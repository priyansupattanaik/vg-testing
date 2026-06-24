import os
import re
import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from cache_utils import setup_cache
from qwen_verifier import QwenFrameVerifier
from settings import cfg
from segmenter import GroundedSegmenter
from tracker import ObjectTracker
from vector_index import SegmentVectorIndex
from video_reader import DecordVideoReader
from vlm import SearchEncoder

setup_cache()


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _write_image(path, frame):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if frame is None or not cv2.imwrite(path, frame):
        raise IOError(f"failed to write image: {path}")
    return path


class VisionGuardPipeline:
    def __init__(self, out_dir=None, yolo=None, clip_model=None, verifier_model=None):
        self.out_dir = out_dir or cfg.out_dir
        yolo = yolo or cfg.yolo_model
        clip_model = clip_model or cfg.clip_model
        verifier_model = verifier_model or cfg.verifier_model
        self.trk = ObjectTracker(model=yolo, conf=cfg.yolo_conf, imgsz=cfg.yolo_imgsz)
        self.enc = SearchEncoder(model=clip_model)
        self.ver = QwenFrameVerifier(model=verifier_model)
        self.seg = GroundedSegmenter(verifier_model=verifier_model, verifier=self.ver)
        self.idx = None
        self.run_dir = None
        self.search_idx = SegmentVectorIndex(bit_width=cfg.index_bit_width)
        self.frame_idx = SegmentVectorIndex(bit_width=cfg.index_bit_width)
        self.pool = ThreadPoolExecutor(max_workers=cfg.index_workers)
        os.makedirs(self.out_dir, exist_ok=True)
        self._warmup_failures = {}
        self._warmup_done = False

    def _color_words(self):
        return (
            "yellow", "white", "black", "gray", "red", "blue", "green", "orange", "brown",
        )

    def _query_colors(self, q):
        q = f" {self._normalize_query(q)} "
        return [x for x in self._color_words() if f" {x} " in q]

    def _estimate_color(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        mx1 = int(cw * 0.275)
        mx2 = int(cw * 0.725)
        my1 = int(ch * 0.275)
        my2 = int(ch * 0.725)
        core = crop[my1:my2, mx1:mx2] if mx2 > mx1 and my2 > my1 else crop
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        hh = hsv[..., 0].astype(np.float32)
        ss = hsv[..., 1].astype(np.float32)
        vv = hsv[..., 2].astype(np.float32)
        valid = vv > 40
        if not valid.any():
            return None
        sat_valid = valid & (ss > 45)
        if sat_valid.any():
            hue = hh[sat_valid]
            blue_ratio = float(((hue >= 95) & (hue <= 130)).mean())
            red_ratio = float(((hue <= 10) | (hue >= 170)).mean())
            green_ratio = float(((hue >= 35) & (hue <= 90)).mean())
            yellow_ratio = float(((hue >= 18) & (hue <= 35)).mean())
            orange_ratio = float(((hue >= 10) & (hue < 18)).mean())
            if blue_ratio >= 0.28:
                return "blue"
            if red_ratio >= 0.28:
                return "red"
            if green_ratio >= 0.28:
                return "green"
            if yellow_ratio >= 0.24:
                return "yellow"
            if orange_ratio >= 0.22:
                return "orange"
        bright = vv[valid]
        sat = ss[valid]
        if bright.mean() > 205 and sat.mean() < 32:
            return "white"
        if bright.mean() < 55:
            return "black"
        if sat.mean() < 24:
            return "gray"
        return None

    def _appearance_tags(self, frame, detections):
        tags = []
        for t in detections:
            name = t["name"]
            if name not in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                continue
            color = self._estimate_color(frame, t["box"])
            if color:
                tags.append(f"{color} {name}")
            tags.append(name)
        return sorted(set(tags))

    def _new_run(self, video):
        name = os.path.splitext(os.path.basename(video))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = os.path.join(self.out_dir, f"{name}_{stamp}")
        if os.path.exists(self.run_dir):
            self.run_dir = os.path.join(self.out_dir, f"{name}_{stamp}_{uuid.uuid4().hex[:8]}")
        for x in ["frames", "reports", "segments"]:
            os.makedirs(os.path.join(self.run_dir, x), exist_ok=True)

    def _preview(self, frame, tracks, ts):
        out = frame.copy()
        for t in tracks[:12]:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 160), 2)
            cv2.putText(out, t["name"], (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 160), 2, cv2.LINE_AA)
        cv2.putText(out, f"{ts:.1f}s", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def _is_non_content_frame(self, frame, tracks):
        if tracks:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float((edges > 0).mean())
        return mean < 40.0 and std < 28.0 and edge_ratio < 0.025

    def _cheap_signature(self, frame, size=(64, 36)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (3, 3), 0)

    def _frame_diff_score(self, sig_a, sig_b):
        if sig_a is None or sig_b is None:
            return 1.0
        diff = cv2.absdiff(sig_a, sig_b)
        return float(diff.mean() / 255.0)

    def _is_interesting_frame(self, frame, prev_sig, ts, last_keep_ts, min_motion=0.025, force_keep_gap=4.0):
        sig = self._cheap_signature(frame)
        if prev_sig is None:
            return True, sig, 1.0, "first"
        score = self._frame_diff_score(sig, prev_sig)
        if score >= min_motion:
            return True, sig, score, "motion"
        if last_keep_ts is None or (ts - last_keep_ts) >= force_keep_gap:
            return True, sig, score, "forced_gap"
        return False, sig, score, "duplicate"

    def _q_objs(self, q):
        q = f" {self._normalize_query(q)} "
        m = {
            "person": [" person ", " people ", " peoples ", " persons ", " man ", " woman ", " human "],
            "car": [" car ", " vehicle ", " sedan "],
            "truck": [" truck ", " lorry "],
            "bus": [" bus "],
            "motorcycle": [" motorcycle ", " motorbike ", " bike ", " scooter "],
            "bicycle": [" bicycle ", " cycle "],
            "backpack": [" backpack ", " bag ", " parcel ", " package "],
            "suitcase": [" suitcase ", " luggage ", " parcel ", " package "],
            "handbag": [" handbag ", " purse ", " parcel ", " package "],
            "umbrella": [" umbrella ", " parasol "],
        }
        out = set()
        for k, rows in m.items():
            if any(x in q for x in rows):
                out.add(k)
        # Keep the query vocabulary aligned with the configured detector instead
        # of limiting object search to the aliases above.
        words = set(q.split())
        singular_words = {word[:-1] for word in words if len(word) > 3 and word.endswith("s")}
        query_words = words | singular_words
        for name in self.trk.names().values():
            label = str(name).strip().lower()
            label_words = label.split()
            if label_words and all(word in query_words for word in label_words):
                out.add(label)
        return sorted(out)

    def _normalize_query(self, q):
        q = q.strip().lower()
        repl = {
            "peoples": "people",
            "persons": "person",
            "human beings": "people",
            "bikes": "bicycle",
            "cycles": "bicycle",
            "cars": "car",
            "trucks": "truck",
            "buses": "bus",
            "umbrellas": "umbrella",
        }
        for src, dst in repl.items():
            q = re.sub(rf"\b{re.escape(src)}\b", dst, q)
        return " ".join(q.split())

    def _query_detector_classes(self, q):
        qobjs = self._q_objs(q)
        if not qobjs:
            return [], {}
        class_ids = self.trk.class_ids(qobjs)
        if not class_ids:
            return [], {}
        names = self.trk.names()
        want = {str(x).strip().lower() for x in qobjs}
        cls_to_name = {int(ci): str(names.get(int(ci), ci)).strip().lower() for ci in class_ids if str(names.get(int(ci), ci)).strip().lower() in want}
        return class_ids, cls_to_name

    def _is_strict_object_query(self, q):
        tokens = set(self._normalize_query(q).split())
        if not tokens:
            return False
        allowed = {
            "person", "people", "man", "woman", "human",
            "car", "truck", "bus", "vehicle", "sedan", "lorry",
            "motorcycle", "motorbike", "bike", "bicycle", "cycle", "scooter",
            "umbrella", "parasol",
            "backpack", "bag", "suitcase", "luggage", "handbag", "purse", "parcel", "package",
            "yellow", "white", "black", "gray", "red", "blue", "green", "orange", "brown",
        }
        return all(x in allowed for x in tokens)

    def _is_simple_unsupported_object_query(self, q):
        tokens = self._normalize_query(q).split()
        if not tokens or self._q_objs(q):
            return False
        color_words = set(self._color_words())
        stop_words = {
            "a", "an", "the", "near", "next", "beside", "behind", "front", "of",
            "on", "in", "at", "with", "without", "left", "right", "top", "bottom",
        }
        content = [x for x in tokens if x not in color_words and x not in stop_words]
        return 0 < len(content) <= 2

    def _matching_detections(self, row, qobjs, qcolors, cls_to_name=None):
        out = []
        wanted = {str(x).strip().lower() for x in qobjs}
        cls_to_name = cls_to_name or {}
        for det in row.get("detections", []):
            name = str(det.get("name", "")).strip().lower()
            if wanted and name not in wanted:
                continue
            color = det.get("color")
            if qcolors and set(qobjs) & {"car", "truck", "bus", "motorcycle", "bicycle"}:
                if not color or color not in qcolors:
                    continue
            if cls_to_name and int(det.get("cls", -1)) in cls_to_name:
                name = cls_to_name[int(det["cls"])]
            out.append({
                "name": name,
                "box": det["box"],
                "conf": det.get("conf", 0.0),
                "cls": det.get("cls"),
                "color": color,
            })
        return out

    def _refine_detector_hits(self, q, top_k):
        class_ids, cls_to_name = self._query_detector_classes(q)
        if not class_ids:
            return []
        qobjs = set(self._q_objs(q))
        qcolors = set(self._query_colors(q))
        if qcolors and not qobjs & {"car", "truck", "bus", "motorcycle", "bicycle"}:
            return []
        rows = []
        for row in self.idx.get("frames", []):
            matched = self._matching_detections(row, qobjs, qcolors, cls_to_name)
            if not matched:
                continue
            best_conf = max(float(x.get("conf", 0.0)) for x in matched)
            if best_conf < 0.2:
                continue
            score = 0.44 + 0.32 * best_conf + 0.05 * max(0, len(matched) - 1)
            rows.append({
                "query": q,
                "score": score,
                "base_score": score,
                "retrieval_mode": "detector",
                "frame_id": row.get("frame_id"),
                "ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": sorted({x["name"] for x in matched}),
                "appearances": row.get("appearances", []),
                "tracks": row["tracks"],
                "matched_detections": matched,
                "det_boxes": [x["box"] for x in matched],
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        if not rows:
            return []
        hits = []
        gap_sec = max(self.idx["meta"]["sample_sec"] * 1.25, 1.0)
        for row in rows:
            if len(hits) >= top_k:
                break
            if any(abs(row["ts"] - x["peak_ts"]) < gap_sec for x in hits):
                continue
            start, end = self._clip_bounds(row["ts"])
            labels = sorted({x["name"] for x in row["matched_detections"]})
            hits.append({
                "query": q,
                "score": row["score"],
                "base_score": row["base_score"],
                "retrieval_mode": "detector",
                "cache_key": f"frame:{row.get('frame_id', row['ts'])}",
                "start": start,
                "end": end,
                "peak_ts": row["ts"],
                "frame_path": row["frame_path"],
                "objects": labels,
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "matched_detections": row["matched_detections"],
                "tags": [],
                "summary": f"detector-matched sampled frame at {row['ts']:.2f}s | detected: {', '.join(labels)}",
            })
        return hits

    def _draw_boxes(self, src_path, boxes, out_name, label_text=None):
        if not src_path or not os.path.exists(src_path):
            return src_path
        frame = cv2.imread(src_path)
        if frame is None:
            return src_path
        h, w = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 120), 2)
        if label_text:
            cv2.putText(frame, label_text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(self.run_dir, "segments", out_name)
        try:
            return _write_image(out_path, frame)
        except OSError:
            return src_path

    def _gallery_fallback_boxes(self, row, query):
        matched = row.get("matched_detections", [])
        if matched:
            return [x["box"] for x in matched if x.get("box")]
        qobjs = set(self._q_objs(query))
        qcolors = set(self._query_colors(query))
        if not qobjs:
            return []
        out = []
        for det in row.get("detections", []):
            if det.get("name") not in qobjs:
                continue
            if qcolors and qobjs & {"car", "truck", "bus", "motorcycle", "bicycle"} and det.get("color") not in qcolors:
                continue
            if det.get("box"):
                out.append(det["box"])
        return out

    def _attach_gallery_frame(self, row, query):
        src = row.get("representative_frame_path") or row.get("frame_path")
        if not src:
            row["gallery_frame"] = src
            return row
        fallback = self._gallery_fallback_boxes(row, query)
        boxes, grounded = self.seg.detect(src, query, fallback_boxes=fallback, frame_key=row.get("cache_key"))
        row["gallery_box_source"] = "grounded" if grounded else "detector" if boxes else "none"
        if boxes:
            stamp = int(round(row.get("peak_ts", row.get("start", 0.0)) * 100))
            kind = "grounded" if grounded else "detector fallback"
            row["gallery_frame"] = self._draw_boxes(src, boxes, f"boxed_match_{row.get('match_id', 0):02d}_{stamp:06d}.jpg", label_text=f"{kind}: {query}")
        else:
            row["gallery_frame"] = src
        return row

    def _query_variants(self, q):
        original = " ".join(q.strip().lower().split())
        ql = self._normalize_query(q)
        out = [original, ql] if original and original != ql else [ql]
        seen = set()
        uniq = []
        for item in out:
            if item not in seen:
                uniq.append(item)
                seen.add(item)
        return uniq

    def _embed_query(self, q):
        vecs = [self.enc.embed_text(x) for x in self._query_variants(q)]
        mix = np.mean(vecs, axis=0).astype(np.float32)
        den = max(np.linalg.norm(mix), 1e-6)
        return mix / den

    def _frame_summary(self, q, peak_ts, objs):
        label = ", ".join(objs) if objs else "no tracked objects"
        return f"best matching sampled frame at {peak_ts:.2f}s | detected: {label}"

    def _clip_bounds(self, ts, pad=None):
        pad = self.idx["meta"]["sample_sec"] if pad is None else pad
        dur = self.idx["meta"]["duration"]
        return max(0.0, ts - pad), min(dur, ts + pad)

    def _reselect_best_frame(self, video_path, start_sec, end_sec, query_vec, step_sec=0.1):
        vr = DecordVideoReader(video_path)
        if len(vr) == 0:
            return None, start_sec, -1.0
        step_frames = max(1, int(round(step_sec * vr.fps)))
        start_idx = max(0, int(round(start_sec * vr.fps)))
        end_idx = min(len(vr) - 1, int(round(end_sec * vr.fps)))
        best_score = -1.0
        best_frame = None
        best_ts = start_sec
        indices = list(range(start_idx, end_idx + 1, step_frames))
        chunk_size = 16
        for offset in range(0, len(indices), chunk_size):
            chunk = indices[offset: offset + chunk_size]
            frames = vr.get_batch(chunk)
            for idx, frame in zip(chunk, frames):
                if frame is None:
                    continue
                emb = self.enc.embed_frame(frame)
                score = float(np.dot(emb, query_vec))
                ts = vr.ts_for(idx)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_ts = ts
        if best_frame is None:
            return None, start_sec, -1.0
        safe_start = str(round(start_sec, 2)).replace(".", "_")
        out_path = os.path.join(self.run_dir, "frames", f"resel_{safe_start}.jpg")
        _write_image(out_path, best_frame)
        return out_path, best_ts, best_score

    def _refresh_det_boxes_for_hit(self, hit, query):
        class_ids, _ = self._query_detector_classes(query)
        if not class_ids:
            return hit.get("det_boxes", [])
        frame_path = hit.get("representative_frame_path") or hit.get("frame_path")
        if not frame_path or not os.path.exists(frame_path):
            return hit.get("det_boxes", [])
        frame = cv2.imread(frame_path)
        if frame is None:
            return hit.get("det_boxes", [])
        want = set(self._q_objs(query))
        dets = self.trk.detect(frame, cls=class_ids, conf=0.12)
        boxes = [det["box"] for det in dets if str(det.get("name", "")).strip().lower() in want]
        return boxes or hit.get("det_boxes", [])

    def _apply_reselection(self, hits, query, query_vec, top_n=1):
        take = min(top_n, len(hits))
        for i in range(take):
            frame_path, best_ts, best_score = self._reselect_best_frame(self.idx["video"], hits[i]["start"], hits[i]["end"], query_vec, step_sec=0.1)
            if not frame_path:
                continue
            hits[i]["representative_frame_path"] = frame_path
            hits[i]["frame_path"] = frame_path
            hits[i]["peak_ts"] = best_ts
            hits[i]["reselected_score"] = best_score
            hits[i]["det_boxes"] = self._refresh_det_boxes_for_hit(hits[i], query)
            if hits[i].get("matched_detections"):
                labels = sorted({x["name"] for x in hits[i]["matched_detections"]})
                hits[i]["summary"] = f"detector-matched sampled frame at {best_ts:.2f}s | detected: {', '.join(labels)}"
        return hits

    def _verification_results(self, rows, query, top_n):
        take = min(top_n, len(rows))
        futures = {
            self.pool.submit(
                self.ver.verify_query,
                rows[i].get("representative_frame_path", rows[i]["frame_path"]),
                query,
                rows[i].get("cache_key"),
            ): i
            for i in range(take)
        }
        for future, idx in futures.items():
            try:
                yield idx, future.result(timeout=cfg.verify_timeout_sec)
            except Exception:
                yield idx, {"matched": False, "confidence": 0.0, "caption": "", "boxes": []}

    def _apply_verification_result(self, row, result, query):
        boxes = result.get("boxes", [])
        caption = result.get("caption", "")
        matched = bool(result.get("matched"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        row["det_boxes"] = boxes or row.get("det_boxes", [])
        row["verified_caption"] = caption
        row["grounded"] = bool(boxes)
        row["verified_match"] = matched
        row["verify_score"] = confidence
        if matched:
            row["score"] = float(row["score"] + min(0.35, 0.16 + 0.18 * confidence))
            label = ", ".join(row.get("objects", [])) or query
            detail = caption or f"visible match for {query}"
            row["summary"] = f"verified query match at {row.get('peak_ts', row['start']):.2f}s | {detail} | detected: {label}"
        elif caption:
            row["score"] = float(row["score"] * 0.6)
            row["low_confidence"] = True
            row["summary"] = f"unverified visual candidate at {row.get('peak_ts', row['start']):.2f}s | {caption}"
        else:
            row["score"] = float(row["score"] * 0.5)
            row["low_confidence"] = True
        return row

    def _verify_rows(self, rows, query, top_n=1):
        if not rows:
            return rows
        for idx, result in self._verification_results(rows, query, top_n):
            self._apply_verification_result(rows[idx], result, query)
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    def _verify_rows_stream(self, rows, query, top_n=1):
        if not rows:
            return
        for idx, result in self._verification_results(rows, query, top_n):
            yield idx, self._apply_verification_result(rows[idx], result, query)

    def _confirmed_rows(self, rows):
        return [x for x in rows if x.get("verified_match")]

    def _cluster_frame_hits(self, rows, top_k, gap_sec):
        rows = sorted(rows, key=lambda x: x["ts"])
        clusters = []
        for row in rows:
            if not clusters or row["ts"] - clusters[-1][-1]["ts"] > gap_sec:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        out = []
        for chunk in clusters:
            peak = max(chunk, key=lambda x: x["score"])
            objs = sorted({obj for row in chunk for obj in row["objects"]})
            start, end = self._clip_bounds(peak["ts"], pad=max(gap_sec, self.idx["meta"]["sample_sec"]))
            out.append({
                "query": peak["query"],
                "score": max(x["score"] for x in chunk),
                "base_score": peak["base_score"],
                "cache_key": f"frame:{peak.get('frame_id', peak['ts'])}",
                "start": start,
                "end": end,
                "peak_ts": peak["ts"],
                "representative_frame_path": peak["frame_path"],
                "frame_path": peak["frame_path"],
                "objects": objs,
                "tracks": sorted({tid for row in chunk for tid in row["tracks"]}),
                "appearances": sorted({tag for row in chunk for tag in row.get("appearances", [])}),
                "det_boxes": [x["box"] for x in peak.get("detections", [])],
                "tags": [],
                "summary": self._frame_summary(peak["query"], peak["ts"], objs),
            })
        out = sorted(out, key=lambda x: x["score"], reverse=True)
        dedup = []
        for row in out:
            if len(dedup) >= top_k:
                break
            if any(abs(row["peak_ts"] - x["peak_ts"]) < gap_sec for x in dedup):
                continue
            dedup.append(row)
        return dedup

    def _fallback_object_hits(self, q, top_k):
        qobjs = set(self._q_objs(q))
        qcolors = set(self._query_colors(q))
        if not qobjs:
            return []
        rows = []
        for row in self.idx.get("frames", []):
            sobj = set(row["objects"])
            hit = len(sobj & qobjs)
            if not hit:
                continue
            appear = set(row.get("appearances", []))
            color_hit = 0
            if qcolors and set(qobjs) & {"car", "truck", "bus", "motorcycle", "bicycle"}:
                for color in qcolors:
                    for obj in qobjs:
                        if f"{color} {obj}" in appear:
                            color_hit += 1
                if color_hit == 0:
                    continue
            score = 0.2 + 0.08 * hit + 0.14 * color_hit
            rows.append({
                "query": q,
                "score": score,
                "base_score": score,
                "retrieval_mode": "object_fallback",
                "frame_id": row.get("frame_id"),
                "ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "det_boxes": [x["box"] for x in row.get("detections", [])],
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        if not rows:
            return []
        hits = []
        for row in rows:
            if len(hits) >= top_k:
                break
            if any(abs(row["ts"] - x["peak_ts"]) < 2.0 for x in hits):
                continue
            start, end = self._clip_bounds(row["ts"])
            hits.append({
                "query": q,
                "score": row["score"],
                "base_score": row["base_score"],
                "retrieval_mode": "object_fallback",
                "cache_key": f"frame:{row.get('frame_id', row['ts'])}",
                "start": start,
                "end": end,
                "peak_ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "det_boxes": row.get("det_boxes", []),
                "tags": [],
                "summary": self._frame_summary(q, row["ts"], row["objects"]),
            })
        for hit in hits:
            appear = ", ".join(hit.get("appearances", []))
            suffix = f" | appearance: {appear}" if appear else ""
            hit["summary"] = f"object-matched sampled frame at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects'])}{suffix}"
            hit["low_confidence"] = True
        return hits

    def index_video_iter(self, video, sample_sec=None, win_sec=None):
        sample_sec = sample_sec or cfg.sample_sec
        win_sec = win_sec or cfg.window_sec
        self._new_run(video)
        self.trk.reset()
        vr = DecordVideoReader(video)
        if len(vr) == 0:
            raise ValueError(f"cannot open video: {video}")
        fps = vr.fps or 25.0
        total = len(vr)
        dur = total / fps if fps else 0.0
        step = max(1, int(round(sample_sec * fps)))
        frames = []
        pending = []
        batch_size = 8
        frame_vec_chunks = []
        frame_id_chunks = []
        prev_sig = None
        last_keep_ts = None
        last_kept_objects = set()
        t0 = time.perf_counter()
        processed_samples = 0
        total_samples = max(1, len(sample_indices := list(range(0, total, step))))

        def flush_pending():
            nonlocal frames, pending, frame_vec_chunks, frame_id_chunks
            if not pending:
                return
            write_futures = [
                self.pool.submit(_write_image, item["frame_path"], item["frame"])
                for item in pending
            ]
            emb_list = self.enc.embed_frames([x["frame"] for x in pending])
            for future in write_futures:
                future.result()
            chunk_vecs = []
            chunk_ids = []
            for item, emb in zip(pending, emb_list):
                frame_id = np.uint64(len(frames))
                frames.append({
                    "frame_id": frame_id,
                    "frame": item["frame_idx"],
                    "ts": item["ts"],
                    "emb": emb,
                    "frame_path": item["frame_path"],
                    "meta": item["meta"],
                })
                chunk_vecs.append(emb)
                chunk_ids.append(frame_id)
            if chunk_vecs:
                frame_vec_chunks.append(np.ascontiguousarray(np.stack(chunk_vecs).astype(np.float32)))
                frame_id_chunks.append(np.asarray(chunk_ids, dtype=np.uint64))
            pending = []

        for offset in range(0, len(sample_indices), batch_size):
            chunk = sample_indices[offset: offset + batch_size]
            batch_frames = vr.get_batch(chunk)
            valid = [(i, frame) for i, frame in zip(chunk, batch_frames) if frame is not None]
            processed_samples += len(chunk)
            if not valid:
                continue
            interesting = []
            for i, frame in valid:
                ts = vr.ts_for(i)
                keep, sig, motion_score, keep_reason = self._is_interesting_frame(frame, prev_sig, ts, last_keep_ts)
                prev_sig = sig
                if keep:
                    interesting.append((i, frame, ts, motion_score, keep_reason))
            if not interesting:
                continue
            det_batches = self.trk.detect_batch([frame for _, frame, _, _, _ in interesting], cls=None, conf=0.18)
            for (i, frame, ts, motion_score, keep_reason), detections in zip(interesting, det_batches):
                if frame is None:
                    continue
                if self._is_non_content_frame(frame, detections):
                    continue
                objs = {}
                det_rows = []
                for det in detections:
                    name = det["name"]
                    objs[name] = objs.get(name, 0) + 1
                    color = None
                    if name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                        color = self._estimate_color(frame, det["box"])
                    det_rows.append({
                        "box": det["box"],
                        "conf": det["conf"],
                        "cls": det["cls"],
                        "name": name,
                        "color": color,
                    })
                meta = {
                    "objects": objs,
                    "tracks": [],
                    "appearances": self._appearance_tags(frame, det_rows),
                    "detections": det_rows,
                    "motion_score": round(float(motion_score), 5),
                    "keep_reason": keep_reason,
                    "object_delta": len(set(objs.keys()) ^ last_kept_objects),
                    "still_people": int(objs.get("person", 0) if motion_score < 0.02 else 0),
                    "person": objs.get("person", 0),
                }
                frame_path = os.path.join(self.run_dir, "frames", f"f_{i:06d}.jpg")
                pending.append({
                    "frame_idx": i,
                    "ts": ts,
                    "frame": frame.copy(),
                    "frame_path": frame_path,
                    "meta": meta,
                })
                last_kept_objects = set(objs.keys())
                last_keep_ts = ts
                elapsed = max(1e-6, time.perf_counter() - t0)
                sample_rate = processed_samples / elapsed
                remain = max(0, total_samples - processed_samples)
                eta = remain / sample_rate if sample_rate > 0 else 0.0
                pct = min(100.0, 100.0 * processed_samples / total_samples)
                status = f"scanning {ts:.1f}s / {dur:.1f}s | {pct:.0f}% | eta {eta:.1f}s"
                yield {"kind": "preview", "image": self._preview(frame, det_rows, ts), "status": status}
                if len(pending) >= self.enc.image_batch_size:
                    flush_pending()
        flush_pending()
        block = max(1, int(round(win_sec / sample_sec)))
        segs = []
        seg_vec_chunks = []
        seg_id_chunks = []
        seg_chunk_vecs = []
        seg_chunk_ids = []
        for j, item in enumerate(frames):
            lo = (j // block) * block
            hi = min(len(frames), lo + block)
            chunk = frames[lo:hi]
            emb = np.mean([x["emb"] for x in chunk], axis=0).astype(np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-6)
            objs = {}
            tids = set()
            motion_scores = []
            still_people = 0
            forced_keeps = 0
            object_delta = 0
            for x in chunk:
                tids |= set(x["meta"]["tracks"])
                for k, v in x["meta"]["objects"].items():
                    objs[k] = max(objs.get(k, 0), v)
                motion_scores.append(float(x["meta"].get("motion_score", 0.0)))
                still_people += int(x["meta"].get("still_people", 0))
                object_delta += int(x["meta"].get("object_delta", 0))
                if x["meta"].get("keep_reason") == "forced_gap":
                    forced_keeps += 1
            segs.append({
                "seg_id": np.uint64(len(segs)),
                "start": chunk[0]["ts"],
                "end": chunk[-1]["ts"],
                "mid": item["ts"],
                "emb": emb,
                "frame_path": item["frame_path"],
                "objects": sorted(objs.keys()),
                "tracks": sorted(tids),
                "temporal_stats": {
                    "avg_motion": round(float(np.mean(motion_scores)) if motion_scores else 0.0, 5),
                    "max_motion": round(float(np.max(motion_scores)) if motion_scores else 0.0, 5),
                    "still_people_frames": still_people,
                    "forced_keep_frames": forced_keeps,
                    "object_delta_sum": object_delta,
                },
                "tags": [],
            })
            seg_chunk_vecs.append(emb)
            seg_chunk_ids.append(np.uint64(len(segs) - 1))
        if seg_chunk_vecs:
            seg_vec_chunks.append(np.ascontiguousarray(np.stack(seg_chunk_vecs).astype(np.float32)))
            seg_id_chunks.append(np.asarray(seg_chunk_ids, dtype=np.uint64))
        meta = {
            "video": video,
            "fps": fps,
            "frames": total,
            "duration": dur,
            "sample_sec": sample_sec,
            "win_sec": win_sec,
            "segments": len(segs),
        }
        self.idx = {
            "video": video,
            "meta": meta,
            "frames": [
                {
                    "frame_id": int(x["frame_id"]),
                    "frame": x["frame"],
                    "ts": x["ts"],
                    "frame_path": x["frame_path"],
                    "representative_frame_path": x["frame_path"],
                    "objects": sorted(x["meta"]["objects"].keys()),
                    "appearances": x["meta"]["appearances"],
                    "tracks": x["meta"]["tracks"],
                    "detections": x["meta"]["detections"],
                    "motion_score": x["meta"].get("motion_score", 0.0),
                    "keep_reason": x["meta"].get("keep_reason", ""),
                    "still_people": x["meta"].get("still_people", 0),
                    "object_delta": x["meta"].get("object_delta", 0),
                }
                for x in frames
            ],
            "segments": segs,
        }
        from collections import Counter

        _obj_counter = Counter()
        for _row in self.idx["frames"]:
            for _obj in _row.get("objects", []):
                _obj_counter[_obj] += 1

        self.idx["meta"]["object_counts"] = dict(_obj_counter.most_common())
        self.idx["meta"]["total_detections"] = sum(_obj_counter.values())
        self.idx["meta"]["unique_objects"] = len(_obj_counter)
        frame_chunks = list(zip(frame_vec_chunks, frame_id_chunks))
        seg_chunks = list(zip(seg_vec_chunks, seg_id_chunks))
        self.frame_idx.build_merged(frame_chunks, path=os.path.join(self.run_dir, "reports", "frame_index.tvim"))
        self.search_idx.build_merged(seg_chunks, path=os.path.join(self.run_dir, "reports", "segment_index.tvim"))
        path = os.path.join(self.run_dir, "reports", "index.json")
        _write_json(
            path,
            {
                "meta": {
                    **self.idx["meta"],
                    "retriever": self.frame_idx.backend,
                    "segment_retriever": self.search_idx.backend,
                    "verifier": self.ver.model_name,
                },
                "frames": [
                    {
                        "frame_id": x["frame_id"],
                        "ts": x["ts"],
                        "frame_path": x["frame_path"],
                        "objects": x["objects"],
                        "appearances": x["appearances"],
                        "detections": x["detections"],
                        "motion_score": x.get("motion_score", 0.0),
                        "keep_reason": x.get("keep_reason", ""),
                        "still_people": x.get("still_people", 0),
                        "object_delta": x.get("object_delta", 0),
                    }
                    for x in self.idx["frames"]
                ],
                "segments": [
                    {
                        "seg_id": int(x["seg_id"]),
                        "start": x["start"],
                        "end": x["end"],
                        "mid": x["mid"],
                        "frame_path": x["frame_path"],
                        "objects": x["objects"],
                        "temporal_stats": x["temporal_stats"],
                        "tags": x["tags"],
                    }
                    for x in segs
                ],
            },
        )
        yield {
            "kind": "done",
                "meta": {
                    **self.idx["meta"],
                    "retriever": self.frame_idx.backend,
                    "segment_retriever": self.search_idx.backend,
                    "verifier": self.ver.model_name,
                },
            "index_json": path,
        }

    def warmup_models(self):
        self._warmup_failures = {}
        for name, fn in [("tracker", self.trk.load),
                         ("encoder", self.enc.load),
                         ("verifier", self.ver.warmup)]:
            try:
                fn()
            except Exception as e:
                self._warmup_failures[name] = str(e)
        self._warmup_done = True

    def warmup_status(self):
        if not self._warmup_done:
            return "Models loading..."
        if not self._warmup_failures:
            return "All models ready."
        return "WARNING: " + " | ".join(
            f"{k} failed: {v}" for k, v in self._warmup_failures.items()
        )

    def _candidate_hits(self, raw_q, top_k=4):
        q = self._normalize_query(raw_q)
        qobjs = self._q_objs(q)
        qcolors = set(self._query_colors(q))
        if self._is_simple_unsupported_object_query(raw_q):
            return q, None, qobjs, [], 0
        qv = self._embed_query(raw_q)
        detailed = not self._is_strict_object_query(q)
        candidate_limit = min(max(top_k * 3, 8), max(8, len(self.idx.get("frames", [])))) if detailed else top_k
        detector_hits = self._refine_detector_hits(q, candidate_limit)
        if detector_hits:
            hits = self._apply_reselection(detector_hits, q, qv, top_n=min(4, len(detector_hits)))
            return q, qv, qobjs, hits, min(8 if detailed else 4, len(hits))
        frames = self.idx.get("frames", [])
        frame_map = {int(x["frame_id"]): x for x in frames}
        fetch_k = min(max(top_k * 12, 36), len(frames))
        frame_scores, frame_ids = self.frame_idx.search(qv, fetch_k)
        rows = []
        for base_score, frame_id in zip(frame_scores, frame_ids):
            row = frame_map.get(int(frame_id))
            if row is None:
                continue
            score = float(base_score)
            sobj = set(row["objects"])
            appear = set(row.get("appearances", []))
            if qobjs:
                hit = len(sobj & set(qobjs))
                if hit:
                    score += 0.1 * hit
                else:
                    score -= 0.08
            if qcolors and set(qobjs) & {"car", "truck", "bus", "motorcycle", "bicycle"}:
                color_hit = 0
                for color in qcolors:
                    for obj in qobjs:
                        if f"{color} {obj}" in appear:
                            color_hit += 1
                if color_hit:
                    score += 0.22 * color_hit
                else:
                    score -= 0.12
            rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "retrieval_mode": "semantic_frame",
                "frame_id": row["frame_id"],
                "ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "appearances": row.get("appearances", []),
                "tracks": row["tracks"],
                "detections": row.get("detections", []),
            })
        ranked_rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        rows = [x for x in ranked_rows if x["score"] >= 0.14]
        out = self._cluster_frame_hits(rows, top_k=top_k, gap_sec=max(self.idx["meta"]["sample_sec"] * 1.25, 1.0))
        if out:
            out = self._apply_reselection(out, q, qv, top_n=min(4, len(out)))
            verify_n = min(8, len(out)) if detailed or not qobjs else min(4, len(out))
            return q, qv, qobjs, out, verify_n
        obj_hits = self._fallback_object_hits(q, top_k)
        if obj_hits:
            obj_hits = self._apply_reselection(obj_hits, q, qv, top_n=min(4, len(obj_hits)))
            return q, qv, qobjs, obj_hits, min(4, len(obj_hits))
        if ranked_rows and not self._is_strict_object_query(q):
            weak = self._cluster_frame_hits(ranked_rows[: max(top_k * 3, 8)], top_k=top_k, gap_sec=max(self.idx["meta"]["sample_sec"] * 1.25, 1.0))
            for hit in weak:
                hit["summary"] = f"low-confidence visual match at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects']) if hit['objects'] else 'no tracked objects'}"
                hit["low_confidence"] = True
                hit["retrieval_mode"] = "weak_semantic"
            if weak:
                weak = self._apply_reselection(weak, q, qv, top_n=min(4, len(weak)))
                verify_n = min(8, len(weak)) if detailed or not qobjs else 1
                return q, qv, qobjs, weak, verify_n
        n = len(self.idx["segments"])
        if n == 0:
            return q, qv, qobjs, [], 0
        seg_map = {int(x["seg_id"]): x for x in self.idx["segments"]}
        fetch_k = min(max(top_k * 8, 24), n)
        base_scores, seg_ids = self.search_idx.search(qv, fetch_k)
        seg_rows = []
        for base_score, seg_id in zip(base_scores, seg_ids):
            seg = seg_map.get(int(seg_id))
            if seg is None:
                continue
            score = float(base_score)
            sobj = set(seg["objects"])
            if qobjs:
                hit = len(sobj & set(qobjs))
                if hit:
                    score += 0.12 * hit
                else:
                    score -= 0.1
            seg_rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "retrieval_mode": "segment",
                "cache_key": f"seg:{int(seg['seg_id'])}",
                "start": seg["start"],
                "end": seg["end"],
                "peak_ts": seg["mid"],
                "representative_frame_path": seg["frame_path"],
                "frame_path": seg["frame_path"],
                "objects": seg["objects"],
                "tracks": seg["tracks"],
                "det_boxes": [],
                "tags": seg["tags"],
                "summary": self._frame_summary(q, seg["mid"], seg["objects"]),
            })
        seg_rows = sorted(seg_rows, key=lambda x: x["score"], reverse=True)
        out = []
        for row in seg_rows:
            if len(out) >= top_k:
                break
            if any(abs(row["peak_ts"] - x["peak_ts"]) < 3 for x in out):
                continue
            if row["score"] < 0.18:
                continue
            out.append(row)
        if out:
            out = self._apply_reselection(out, q, qv, top_n=min(4, len(out)))
            verify_n = min(8, len(out)) if detailed or not qobjs else min(4, len(out))
            return q, qv, qobjs, out, verify_n
        return q, qv, qobjs, [], 0

    def search_stream(self, raw_q, top_k=4):
        q, _, qobjs, candidates, verify_n = self._candidate_hits(raw_q, top_k=top_k)
        if not candidates:
            yield []
            return
        if not self.ver.backend or self.ver.backend == "none":
            for _ in range(30):
                if self.ver.backend not in (None, "none"):
                    break
                time.sleep(1)
        if self.ver.backend == "dev_passthrough":
            yield self._unverified_rows(candidates, "visual verifier unavailable on Windows CPU")[:top_k]
            return
        working = [dict(x) for x in candidates]
        confirmed = []
        emitted = set()
        for idx, row in self._verify_rows_stream(working, q, top_n=verify_n):
            if row.get("verified_match"):
                confirmed = sorted(self._confirmed_rows(working), key=lambda x: x["score"], reverse=True)[:top_k]
                key = tuple(x.get("cache_key") for x in confirmed)
                if key not in emitted:
                    emitted.add(key)
                    yield confirmed
        if not emitted:
            trusted = [x for x in working if x.get("retrieval_mode") in {"detector", "object_fallback"}]
            if qobjs and trusted:
                yield sorted(trusted, key=lambda x: x["score"], reverse=True)[:top_k]
            else:
                yield []

    def search(self, q, top_k=4):
        checked_q, _, qobjs, candidates, verify_n = self._candidate_hits(q.strip(), top_k=top_k)
        if not candidates:
            return []
        if not self.ver.backend or self.ver.backend == "none":
            for _ in range(30):
                if self.ver.backend not in (None, "none"):
                    break
                time.sleep(1)
        if self.ver.backend == "dev_passthrough":
            return self._unverified_rows(candidates, "visual verifier unavailable on Windows CPU")[:top_k]
        checked = self._verify_rows(candidates, checked_q, top_n=verify_n)
        confirmed = self._confirmed_rows(checked)[:top_k]
        if confirmed:
            return confirmed
        trusted = [x for x in checked if x.get("retrieval_mode") in {"detector", "object_fallback"}]
        if qobjs and trusted:
            return sorted(trusted, key=lambda x: x["score"], reverse=True)[:top_k]
        return []

    def _unverified_rows(self, rows, reason):
        out = []
        for row in rows:
            item = dict(row)
            item["low_confidence"] = True
            item["verified_match"] = False
            item["summary"] = f"unverified visual candidate at {item.get('peak_ts', item['start']):.2f}s | {reason}"
            out.append(item)
        return sorted(out, key=lambda x: x["score"], reverse=True)

    def prepare_hits(self, hits, query):
        out = []
        for i, hit in enumerate(hits, 1):
            row = dict(hit)
            row["match_id"] = i
            row["label"] = f"{i}. {hit.get('peak_ts', hit['start']):.2f}s"
            row["representative_frame_path"] = row.get("representative_frame_path") or row.get("frame_path")
            row["gallery_frame"] = row["representative_frame_path"]
            row = self._attach_gallery_frame(row, query)
            out.append(row)
        return out


