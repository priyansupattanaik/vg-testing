import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from qwen_verifier import QwenFrameVerifier


class GroundedSegmenter:
    def __init__(self, sam="facebook/sam2.1-hiera-small", verifier_model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", verifier=None, device=None):
        self.sam_name = sam
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ver = verifier or QwenFrameVerifier(model=verifier_model, device=self.dev)
        self.sp = None
        self.sm = None

    def load(self):
        if self.sm is None:
            from transformers import Sam2Model, Sam2Processor

            self.sp = Sam2Processor.from_pretrained(self.sam_name)
            self.sm = Sam2Model.from_pretrained(self.sam_name, device_map="auto" if self.dev == "cuda" else None)
            if self.dev != "cuda":
                self.sm.to(self.dev)

    def detect(self, frame_path, query, fallback_boxes=None):
        boxes = self.ver.ground_phrase(frame_path, query.strip().lower())
        if not boxes:
            boxes = fallback_boxes or []
        scores = [max(0.15, 1.0 - 0.08 * i) for i in range(len(boxes))]
        texts = [query.strip().lower()] * len(boxes)
        return boxes, scores, texts

    def segment(self, frame, boxes):
        self.load()
        if not boxes:
            return []
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.sp(images=img, input_boxes=[boxes], return_tensors="pt").to(self.sm.device)
        with torch.no_grad():
            out = self.sm(**inp, multimask_output=False)
        masks = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        res = []
        for i in range(masks.shape[0]):
            mask = masks[i, 0].numpy() if masks.ndim == 4 else masks[i].numpy()
            res.append(mask > 0)
        return res

    def overlay(self, frame, boxes, scores, masks):
        out = frame.copy()
        cols = [(40, 220, 120), (220, 120, 40), (60, 140, 240), (220, 60, 170)]
        for i, box in enumerate(boxes):
            c = cols[i % len(cols)]
            x1, y1, x2, y2 = [int(v) for v in box]
            if i < len(masks):
                m = masks[i]
                lay = np.zeros_like(out)
                lay[m] = c
                out = cv2.addWeighted(out, 1.0, lay, 0.35, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            cv2.putText(out, f"{scores[i]:.2f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        return out

    def segment_clip(self, video, query, out_path, frame_dir, stride=3, fallback_boxes=None):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        tmp = f"{out_path}.part.mp4"
        if os.path.exists(tmp):
            os.remove(tmp)
        out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        prev = None
        picks = []
        seen = 0
        first = None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            if i % stride == 0:
                frame_path = os.path.join(frame_dir, f"segment_src_{i:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                boxes, scores, _ = self.detect(frame_path, query, fallback_boxes=fallback_boxes)
                masks = self.segment(frame, boxes[:2]) if boxes else []
                prev = self.overlay(frame, boxes[:2], scores[:2], masks)
                if boxes:
                    seen += 1
                    p = os.path.join(frame_dir, f"seg_{i:05d}.jpg")
                    cv2.imwrite(p, prev)
                    picks.append(p)
            out.write(prev if prev is not None else frame)
            i += 1
        cap.release()
        out.release()
        self._finalize_video(tmp, out_path)
        if seen == 0 and first is not None:
            p = os.path.join(frame_dir, "fallback_00000.jpg")
            cv2.imwrite(p, first)
            picks.append(p)
        return out_path, picks, seen

    def _finalize_video(self, tmp, out_path):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            conv = f"{out_path}.conv.mp4"
            if os.path.exists(conv):
                os.remove(conv)
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                tmp,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                conv,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(conv, out_path)
                os.remove(tmp)
                return
            except Exception:
                if os.path.exists(conv):
                    os.remove(conv)
        os.replace(tmp, out_path)
