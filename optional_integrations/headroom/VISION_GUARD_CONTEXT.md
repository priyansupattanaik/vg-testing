# Vision Guard Context Snapshot

This file is a lightweight repository snapshot for optional external compression or agent handoff workflows. It is not imported by the Vision Guard runtime.

## Repository Identity

- Project name: `Vision Guard`
- Audited workspace path: `D:\CDAC_PROJECT\5.CV_Project`
- Runtime style: Python inference app with Gradio UI
- Training loop present: no

## Runtime Modules

- `app.py`
- `pipeline.py`
- `cache_utils.py`
- `clip_generator.py`
- `qwen_verifier.py`
- `report_generator.py`
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
- Segmentation: SAM2.1 Hiera Small
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
3. reject event-style queries
4. run detector-first and semantic retrieval
5. reselect dense best frames
6. verify top candidates with Qwen in parallel
7. cache verifier results by query and frame key

### Export path

1. resolve selected hits
2. generate or reuse raw clips
3. segment selected clips with SAM2 using Qwen or detector boxes
4. write JSON, CSV, HTML, and ZIP outputs

## Current Notable Constraints

- event-style queries are intentionally disabled
- unsupported simple exact-object labels are rejected conservatively
- Windows CPU uses a verifier development bypass rather than full Qwen inference
- Headroom is not part of the runtime path

## Best Companion Files

For a fuller downstream handoff, pair this snapshot with:

- `PROJECT_DOCUMENTATION.md`
- `README.md`

## Removal

This file is optional. Deleting it does not affect application behavior.
