import os

import numpy as np

try:
    from turbovec import IdMapIndex
except Exception:
    IdMapIndex = None


class SegmentVectorIndex:
    def __init__(self, bit_width=4):
        self.bit_width = bit_width
        self.idx = None
        self.ids = np.zeros((0,), dtype=np.uint64)
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        self.path = None
        self.backend = "numpy"

    def build(self, vectors, ids, path=None):
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2D float32 array")
        ext_ids = np.asarray(ids, dtype=np.uint64)
        if arr.shape[0] != ext_ids.shape[0]:
            raise ValueError("vectors and ids length mismatch")
        self.vecs = arr
        self.ids = ext_ids
        self.path = path
        self.idx = None
        self.backend = "numpy"
        if IdMapIndex is None or arr.size == 0:
            return
        try:
            self.idx = IdMapIndex(dim=arr.shape[1], bit_width=self.bit_width)
            self.idx.add_with_ids(np.ascontiguousarray(arr), ext_ids)
            self.idx.prepare()
            self.backend = "turbovec"
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.idx.write(path)
        except Exception:
            self.idx = None
            self.backend = "numpy"

    def build_merged(self, chunks, path=None):
        vec_parts = []
        id_parts = []
        for vecs, ids in chunks:
            arr = np.asarray(vecs, dtype=np.float32)
            ext_ids = np.asarray(ids, dtype=np.uint64)
            if arr.size == 0 or ext_ids.size == 0:
                continue
            if arr.ndim != 2:
                raise ValueError("chunk vectors must be a 2D float32 array")
            if arr.shape[0] != ext_ids.shape[0]:
                raise ValueError("chunk vectors and ids length mismatch")
            vec_parts.append(arr)
            id_parts.append(ext_ids)
        if not vec_parts:
            self.build(np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.uint64), path=path)
            return
        merged_vecs = np.ascontiguousarray(np.concatenate(vec_parts, axis=0))
        merged_ids = np.ascontiguousarray(np.concatenate(id_parts, axis=0))
        self.build(merged_vecs, merged_ids, path=path)

    def search(self, query, k):
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.backend == "turbovec" and self.idx is not None and len(self.ids):
            scores, ids = self.idx.search(np.ascontiguousarray(q), k=k)
            return np.asarray(scores[0], dtype=np.float32), np.asarray(ids[0], dtype=np.uint64)
        if not len(self.ids):
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.uint64)
        sims = self.vecs @ q[0]
        order = np.argsort(-sims)[:k]
        return sims[order].astype(np.float32), self.ids[order]
