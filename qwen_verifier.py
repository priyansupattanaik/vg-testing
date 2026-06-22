import json
import os
import re
import threading
import platform

import torch
from PIL import Image

_DEV_MODE = platform.system() == "Windows" and not torch.cuda.is_available()


class QwenFrameVerifier:
    _ABSTRACT_TERMS = frozenset({
        "fight","fighting","assault","brawl",
        "fall","falling","collapse",
        "collision","collide","crash","accident",
        "crowd","crowded","gathering",
        "loitering","loiter","suspicious","violence"
    })

    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.process_vision_info = None
        self.vllm_engine = None
        self.vllm_sampling = None
        self.backend = "none"
        self.failed = False
        self.cache = {}
        self.lock = threading.Lock()

    def _confidence_threshold(self, query: str) -> float:
        if any(t in query.lower() for t in self._ABSTRACT_TERMS):
            return 0.30
        return 0.45

    def load(self):
        if _DEV_MODE:
            self.backend = "dev_passthrough"
            return
        if self.model is not None or self.vllm_engine is not None or self.failed:
            return
        if self._load_vllm():
            return
        self._load_hf()

    def _load_vllm(self):
        if self.dev != "cuda":
            return False
        try:
            from vllm import LLM, SamplingParams

            self.vllm_engine = LLM(
                self.model_name,
                trust_remote_code=True,
                dtype="half",
                limit_mm_per_prompt={"image": 1},
            )
            self.vllm_sampling = SamplingParams(temperature=0.0, max_tokens=100)
            self.backend = "vllm"
            return True
        except Exception:
            self.vllm_engine = None
            self.vllm_sampling = None
            return False

    def _load_hf(self):
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map="auto" if self.dev == "cuda" else None,
                trust_remote_code=True,
            )
            if self.dev != "cuda":
                self.model.to(self.dev)
            self.model.eval()
            self.process_vision_info = process_vision_info
            self.backend = "hf"
        except Exception:
            self.failed = True
            self.model = None
            self.processor = None
            self.process_vision_info = None

    def _extract_json(self, text):
        if not text:
            return {}
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    def _clean_boxes(self, boxes, size):
        w, h = size
        clean = []
        for box in boxes or []:
            if isinstance(box, dict):
                box = box.get("box") or box.get("bbox") or box.get("coordinates")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            vals = [float(x) for x in box]
            if max(vals) <= 1.5:
                vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
            if max(vals) <= 1000.0 and (vals[2] > w or vals[3] > h):
                vals = [vals[0] / 1000.0 * w, vals[1] / 1000.0 * h, vals[2] / 1000.0 * w, vals[3] / 1000.0 * h]
            x1, y1, x2, y2 = vals
            x1 = max(0.0, min(w, x1))
            x2 = max(0.0, min(w, x2))
            y1 = max(0.0, min(h, y1))
            y2 = max(0.0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            clean.append([x1, y1, x2, y2])
        return clean

    def _ask(self, frame_path, prompt, max_new_tokens=100):
        self.load()
        if self.vllm_engine is not None:
            return self._ask_vllm(frame_path, prompt, max_new_tokens=max_new_tokens)
        if self.model is None or self.processor is None or self.process_vision_info is None:
            return ""
        if not frame_path or not os.path.exists(frame_path):
            return ""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = generated[:, inputs.input_ids.shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except Exception:
            return ""

    def _ask_vllm(self, frame_path, prompt, max_new_tokens=100):
        if self.vllm_engine is None or not frame_path or not os.path.exists(frame_path):
            return ""
        image = Image.open(frame_path).convert("RGB")
        sampling = self.vllm_sampling
        if sampling is None or getattr(sampling, "max_tokens", None) != max_new_tokens:
            try:
                from vllm import SamplingParams

                sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
            except Exception:
                sampling = None
        if sampling is None:
            return ""
        req = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
        try:
            outputs = self.vllm_engine.generate([req], sampling)
            if not outputs:
                return ""
            out = outputs[0].outputs
            if not out:
                return ""
            return str(out[0].text).strip()
        except Exception:
            return ""

    def warmup(self):
        self.load()
        return not self.failed

    def _cache_key(self, frame_path, query, frame_key=None):
        norm_q = " ".join(query.strip().lower().split())
        return ("verify", frame_key or frame_path, norm_q)

    def verify_query(self, frame_path, query, frame_key=None):
        if self.backend == "dev_passthrough":
            return {
                "matched": True,
                "confidence": 0.55,
                "caption": "[dev mode — Qwen skipped on Windows CPU]",
                "boxes": []
            }
        key = self._cache_key(frame_path, query, frame_key=frame_key)
        if key in self.cache:
            return dict(self.cache[key])
        image = Image.open(frame_path).convert("RGB") if frame_path and os.path.exists(frame_path) else None
        if image is None:
            return {"matched": False, "confidence": 0.0, "caption": "", "boxes": []}
        prompt = (
            "You are verifying CCTV search results. "
            "Decide whether the image clearly satisfies the exact user query. "
            "Treat the query literally and conservatively. "
            "Do not substitute a similar object class for the exact queried class. "
            "For example, a generic car is not a taxi unless explicit taxi markings or signage are visible. "
            "A truck is not a bus. A handbag is not a backpack. "
            "Return matched=true only when the queried object or phrase is visibly present and localizable in the frame. "
            "If matched=true, provide at least one tight box around the visible matching region. "
            "Do not infer beyond the visible evidence. "
            f"User query: {query}\n"
            "Return JSON only with keys: matched(boolean), confidence(number 0 to 1), "
            "description(short string), boxes(list of [x1,y1,x2,y2] pixel boxes for visible matching regions, empty if not localizable)."
        )
        raw = self._ask(frame_path, prompt)
        data = self._extract_json(raw)
        boxes = self._clean_boxes(data.get("boxes", []), image.size)
        matched = bool(data.get("matched", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        if matched and not boxes:
            matched = False
        if confidence < self._confidence_threshold(query):
            matched = False
        result = {
            "matched": matched,
            "confidence": max(0.0, min(1.0, confidence)),
            "caption": str(data.get("description", "") or "").strip(),
            "boxes": boxes,
        }
        with self.lock:
            self.cache[key] = dict(result)
        return result

    def ground_phrase(self, frame_path, phrase, multi=True, frame_key=None):
        result = self.verify_query(frame_path, phrase, frame_key=frame_key)
        return result.get("boxes", []) if result.get("matched") else []
