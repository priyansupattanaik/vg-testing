import os
import shutil

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".yolo"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model="yolo11m.pt", conf=0.22, imgsz=640, tracker="botsort.yaml", device=None):
        self.model_name = model
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = self.dev == "cuda"
        self.m = None

    def reset(self):
        self.m = None

    def _cached_model_path(self):
        if os.path.dirname(self.model_name):
            return self.model_name
        base = "/content/drive/MyDrive/visionguard_cache/ultralytics/weights"
        if not os.path.exists("/content/drive/MyDrive"):
            return self.model_name
        os.makedirs(base, exist_ok=True)
        cached = os.path.join(base, self.model_name)
        return cached if os.path.exists(cached) else self.model_name

    def load(self):
        if self.m is not None:
            return
        model_path = self._cached_model_path()
        self.m = YOLO(model_path)
        if model_path == self.model_name and os.path.exists(self.model_name) and os.path.exists("/content/drive/MyDrive"):
            cached = os.path.join("/content/drive/MyDrive/visionguard_cache/ultralytics/weights", self.model_name)
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            if not os.path.exists(cached):
                shutil.copy2(self.model_name, cached)
        self.m.to(self.dev)

    def class_ids(self, names):
        self.load()
        want = {str(x).strip().lower() for x in names}
        out = []
        for ci, name in self.m.names.items():
            if str(name).strip().lower() in want:
                out.append(int(ci))
        return out

    def names(self):
        self.load()
        return {int(k): str(v) for k, v in self.m.names.items()}

    def track(self, frame, cls=None):
        self.load()
        res = self.m.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            half=self.use_half,
            tracker=self.tracker,
            classes=cls,
        )
        out = []
        for r in res:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for tid, box, cf, ci in zip(ids, boxes, confs, clss):
                out.append({
                    "id": int(tid),
                    "box": [round(x, 2) for x in box],
                    "conf": round(float(cf), 4),
                    "cls": int(ci),
                    "name": self.m.names.get(int(ci), str(ci)),
                })
        return out

    def detect(self, frame, cls=None, conf=None):
        self.load()
        res = self.m.predict(
            frame,
            verbose=False,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            half=self.use_half,
            classes=cls,
        )
        out = []
        for r in res:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for box, cf, ci in zip(boxes, confs, clss):
                out.append({
                    "box": [round(x, 2) for x in box],
                    "conf": round(float(cf), 4),
                    "cls": int(ci),
                    "name": self.m.names.get(int(ci), str(ci)),
                })
        return out

    def detect_batch(self, frames, cls=None, conf=None):
        self.load()
        if not frames:
            return []
        res = self.m.predict(
            frames,
            verbose=False,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            half=self.use_half,
            classes=cls,
        )
        batch = []
        for r in res:
            rows = []
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                for box, cf, ci in zip(boxes, confs, clss):
                    rows.append({
                        "box": [round(x, 2) for x in box],
                        "conf": round(float(cf), 4),
                        "cls": int(ci),
                        "name": self.m.names.get(int(ci), str(ci)),
                    })
            batch.append(rows)
        return batch
