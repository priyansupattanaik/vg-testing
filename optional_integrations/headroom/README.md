# Headroom Integration Notes

This folder is intentionally isolated from the main Vision Guard runtime.

It is preserved as optional documentation only. The current application does not import Headroom from:

- `app.py`
- `pipeline.py`
- `qwen_verifier.py`
- `segmenter.py`
- `tracker.py`
- `vlm.py`

That means removing this folder would not change the runtime behavior of the application itself. It remains in the repository only because the main documentation explicitly references it as an optional external context-compression scaffold.

## Current Status

- Runtime-active: no
- Required for the app to launch: no
- Referenced by repository documentation: yes
- Safe to treat as optional: yes

Current project context snapshot:

- [VISION_GUARD_CONTEXT.md](VISION_GUARD_CONTEXT.md)

## Intended Purpose

Headroom is not part of the CCTV retrieval pipeline. It is an external context-management idea for:

- compressing large reports before handing them to another LLM or agent
- compressing long debug logs
- compressing multi-run summaries for downstream tooling

It is not a substitute for:

- YOLO detection
- SigLIP2 retrieval
- Qwen verification
- detector-box fallback gallery rendering

## Recommended Integration Boundary

If Headroom is used later, keep it outside the Vision Guard runtime and apply it only after Vision Guard has already produced:

- `index.json`
- other large text artifacts intended for external LLM workflows

## Removal

This folder is optional documentation only. If the repository no longer wants to carry optional Headroom notes, delete:

- `optional_integrations/headroom/README.md`
- `optional_integrations/headroom/VISION_GUARD_CONTEXT.md`

No runtime code changes are required unless Headroom is wired into the application in the future.
