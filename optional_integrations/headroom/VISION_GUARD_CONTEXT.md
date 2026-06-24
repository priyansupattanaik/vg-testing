# Vision Guard Context Snapshot

This file is a lightweight repository snapshot for optional external compression or agent handoff workflows. It is not imported by the Vision Guard runtime.

## Repository Identity

- Project name: `Vision Guard`
- Workspace path: `D:\CDAC_PROJECT\vg-testing`
- Runtime style: Python inference app with Gradio UI
- Training loop present: no

## Runtime Modules

- `app.py`
- `pipeline.py`
- `cache_utils.py`
- `qwen_verifier.py`
- `segmenter.py`
- `tracker.py`
- `vector_index.py`
- `video_reader.py`
- `vlm.py`

## Current Runtime Stack

- UI: Gradio
- Video access: Decord with OpenCV fallback
- Detector metadata: Ultralytics YOLO
- Retrieval embedding model: SigLIP2 So400m
- Verification and grounding: Qwen2.5-VL-7B-Instruct-AWQ
- Gallery grounding: Qwen verifier with detector-box fallback
- Vector index: turbovec with NumPy fallback

## Current Behavioral Summary

### Scan path

1. sample frames from the video
2. reject duplicates / low-information frames
3. run YOLO batch detection
4. persist frame images
5. embed frames with SigLIP2
6. build frame and segment indexes
7. aggregate object counts into scan metadata

### Query path

1. normalize the query
2. derive supported object and color hints
3. run detector-first and semantic retrieval
4. reselect dense best frames
5. verify top candidates with Qwen in parallel
6. cache verifier results by query and frame key

On Windows CPU development mode, Qwen verification is unavailable. The query path returns explicitly low-confidence semantic candidates instead of claiming a verified match.

### Gallery path

1. prepare confirmed hits
2. reuse verifier results through the query and frame cache key
3. draw grounded boxes on gallery images

## Current Notable Constraints

- detailed natural-language queries use semantic retrieval and Qwen verification
- vehicle paint tags apply only to vehicle queries; clothing-color phrases use semantic retrieval
- unsupported simple exact-object labels are rejected conservatively
- Windows CPU uses a verifier development bypass rather than full Qwen inference
- Headroom is not part of the runtime path

## Best Companion Files

For a fuller downstream handoff, pair this snapshot with:

- `PROJECT_DOCUMENTATION.md`
- `README.md`

## Removal

This file is optional. Deleting it does not affect application behavior.
