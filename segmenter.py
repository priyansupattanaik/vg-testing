from qwen_verifier import QwenFrameVerifier


class GroundedSegmenter:
    def __init__(self, verifier_model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", verifier=None, device=None):
        self.ver = verifier or QwenFrameVerifier(model=verifier_model, device=device)

    def detect(self, frame_path, query, fallback_boxes=None, frame_key=None):
        grounded = self.ver.ground_phrase(frame_path, query.strip().lower(), frame_key=frame_key)
        boxes = grounded
        if not boxes:
            boxes = fallback_boxes or []
        scores = [max(0.15, 1.0 - 0.08 * i) for i in range(len(boxes))]
        texts = [query.strip().lower()] * len(boxes)
        return boxes, scores, texts, bool(grounded)
