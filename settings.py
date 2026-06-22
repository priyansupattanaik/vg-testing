import os


def _int(name, default, low=1):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= low else default


def _float(name, default, low=0.0):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= low else default


def _text(name, default):
    return os.getenv(name, "").strip() or default


class Settings:
    def __init__(self):
        self.out_dir = _text("VISION_GUARD_OUTPUT_DIR", "output")
        self.yolo_model = _text("VISION_GUARD_YOLO_MODEL", "yolo11m.pt")
        self.clip_model = _text("VISION_GUARD_CLIP_MODEL", "google/siglip2-so400m-patch14-384")
        self.verifier_model = _text("VISION_GUARD_VERIFIER_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        self.yolo_conf = _float("VISION_GUARD_YOLO_CONF", 0.22)
        self.yolo_imgsz = _int("VISION_GUARD_YOLO_IMGSZ", 640)
        self.index_workers = _int("VISION_GUARD_INDEX_WORKERS", 4)
        self.index_bit_width = _int("VISION_GUARD_INDEX_BIT_WIDTH", 4)
        self.sample_sec = _float("VISION_GUARD_SAMPLE_SEC", 0.75)
        self.window_sec = _float("VISION_GUARD_WINDOW_SEC", 4.5)
        self.query_top_k = _int("VISION_GUARD_QUERY_TOP_K", 4)
        self.gallery_columns = _int("VISION_GUARD_GALLERY_COLUMNS", 2)
        self.verify_timeout_sec = _int("VISION_GUARD_VERIFY_TIMEOUT_SEC", 30)


cfg = Settings()
