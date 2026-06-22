import cv2

import numpy as np
import torch
from PIL import Image


class SearchEncoder:
    def __init__(self, model="google/siglip2-so400m-patch14-384", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_batch_size = self._default_image_batch_size()
        self.p = None
        self.m = None
        self.compiled = False

    def _default_image_batch_size(self):
        if self.dev != "cuda":
            return 8
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
        except Exception:
            gpu_name = ""
        return 32 if "a100" in gpu_name else 16

    def load(self):
        if self.m is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self.p = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.m = AutoModel.from_pretrained(self.model_name, dtype=dtype, device_map=None)
        self.m.to(self.dev)
        self.m.eval()
        self._maybe_compile()

    def _maybe_compile(self):
        # torch.compile with CUDA graphs is incompatible with Gradio's
        # worker-thread execution model (TLS assertion failure at runtime).
        # SigLIP2-So400m is fast enough on GPU without compilation.
        self.compiled = False

    def _vec(self, x):
        if hasattr(x, "pooler_output"):
            return x.pooler_output
        if hasattr(x, "image_embeds"):
            return x.image_embeds
        if hasattr(x, "text_embeds"):
            return x.text_embeds
        return x

    def _norm(self, x):
        x = self._vec(x).detach().cpu().numpy()[0]
        n = np.linalg.norm(x)
        if n == 0:
            return x.astype(np.float32)
        return (x / n).astype(np.float32)

    def embed_text(self, txt):
        self.load()
        txt = f"this is a photo of {txt.strip().lower()}"
        inp = self.p(text=[txt], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_text_features(**inp)
        return self._norm(vec)

    def embed_frame(self, frame):
        self.load()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.p(images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_image_features(**inp)
        return self._norm(vec)

    def embed_frames(self, frames):
        self.load()
        if not frames:
            return []
        out = []
        for offset in range(0, len(frames), self.image_batch_size):
            batch = frames[offset: offset + self.image_batch_size]
            imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch]
            inp = self.p(images=imgs, return_tensors="pt").to(self.dev)
            if self.dev == "cuda":
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    vecs = self._vec(self.m.get_image_features(**inp)).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    vecs = self._vec(self.m.get_image_features(**inp)).detach().cpu().numpy()
            for vec in vecs:
                n = np.linalg.norm(vec)
                if n == 0:
                    out.append(vec.astype(np.float32))
                else:
                    out.append((vec / n).astype(np.float32))
        return out
