import os
import threading
import warnings
from pathlib import Path

import gradio as gr

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline
from settings import cfg

setup_cache()
warnings.filterwarnings("ignore", message="The 'theme' parameter in the Blocks constructor will be removed in Gradio 6\\.0.*")
warnings.filterwarnings("ignore", message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6\\.0.*")

ROOT = Path(__file__).resolve().parent
pipe = VisionGuardPipeline()
threading.Thread(target=pipe.warmup_models, daemon=True).start()

theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")
css = """
.gradio-container{max-width:1240px!important}
.gradio-container,.gradio-container *{box-sizing:border-box}
.hero{padding:22px 24px;border-radius:22px;background:linear-gradient(135deg,#16364a 0%,#1f6d78 60%,#b7d9c8 100%);color:#fff;margin-bottom:16px}
.hero h1{margin:0 0 6px 0;font-size:34px}
.hero p{margin:0;font-size:15px;opacity:.96}
.app-shell{gap:18px}
.panel{border:1px solid #253043;border-radius:18px;background:#111827;padding:14px}
.tight-md{margin-top:8px}
.tight-md p{margin:0}
.result-stack{gap:14px}
.hidden-empty{min-height:0!important}
"""


def _in_colab():
    return bool(
        os.getenv("COLAB_RELEASE_TAG")
        or os.getenv("COLAB_BACKEND_VERSION")
        or os.getenv("COLAB_GPU")
        or (os.getenv("JPY_PARENT_PID") and str(ROOT).startswith("/content/"))
    )


def _server_name():
    override = os.getenv("VISION_GUARD_HOST", "").strip()
    if override:
        return override
    if _in_colab() or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "0.0.0.0"
    return "127.0.0.1"


def _share_enabled():
    raw = os.getenv("GRADIO_SHARE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return False


def _sample_videos():
    assets = ROOT / "assets"
    if not assets.exists():
        return []
    return [str(p) for p in sorted(assets.glob("*.mp4"))]


def _meta(meta):
    out = (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- indexed windows: `{meta['segments']}`\n"
        f"- retriever: `{meta.get('retriever', 'numpy')}`\n"
        f"- verifier: `{meta.get('verifier', 'none')}`"
    )
    counts = meta.get("object_counts", {})
    if counts:
        lines = [f"**{name}**: {n}" for name, n in counts.items()]
        obj_md = "  ".join(lines)
        out += "\n\n**Objects detected:**\n" + obj_md
    return out


def _ans(q, rows):
    out = [f"## answer for `{q}`", ""]
    if not rows:
        out.append("no strong matches found")
        return "\n".join(out)
    for i, x in enumerate(rows, 1):
        out.append(f"{i}. `best frame {x.get('peak_ts', x['start']):.2f}s | clip {x['start']:.2f}s - {x['end']:.2f}s`")
        prefix = "low confidence: " if x.get("low_confidence") else ""
        out.append(f"   {prefix}{x['summary']}")
    return "\n".join(out)


def _gallery(rows):
    out = []
    for i, x in enumerate(rows):
        frame_path = x.get("gallery_frame")
        frame_path = frame_path or x.get("frame_path")
        if not frame_path:
            continue
        prefix = "low confidence | " if x.get("low_confidence") else ""
        source = x.get("gallery_box_source")
        box_note = "grounded box | " if source == "grounded" else "detector fallback box | " if source == "detector" else "not localized | "
        caption = f"{x['query']} | {x.get('peak_ts', x['start']):.2f}s | {box_note}{prefix}{x['summary']}" if i == 0 else f"{x['label']} | {box_note}{prefix}{x['summary']}"
        out.append((frame_path, caption))
    return out


def scan_only(video):
    if not video:
        yield "upload a video first", None, "", gr.update(interactive=False), gr.update(interactive=False)
        return
    yield "starting scan", None, "", gr.update(interactive=False), gr.update(interactive=False)
    meta = None
    try:
        for ev in pipe.index_video_iter(video):
            if ev["kind"] == "preview":
                yield ev["status"], ev["image"], "", gr.update(interactive=False), gr.update(interactive=False)
            else:
                meta = ev.get("meta")
    except Exception as exc:
        yield f"scan failed: {exc}", None, "", gr.update(interactive=False), gr.update(interactive=False)
        return
    if not meta:
        yield "scan failed: no index metadata was produced", None, "", gr.update(interactive=False), gr.update(interactive=False)
        return
    yield "scan complete", None, _meta(meta), gr.update(interactive=True), gr.update(interactive=True)


def _find_payload(status, q, seg):
    rows = [[round(x.get("peak_ts", x["start"]), 2), f"{x['start']:.2f}s - {x['end']:.2f}s", ", ".join(x["objects"]), x["summary"]] for x in seg]
    ans = _ans(q.strip(), seg)
    gal = _gallery(seg)
    if not seg:
        note = "### matched frames\n\nNo strong frame matches were found for this query."
    elif any(x.get("low_confidence") for x in seg):
        note = "### matched frames\n\nThe gallery below shows the nearest available sampled frames. These results are low confidence, so review them carefully."
    else:
        note = "### matched frames\n\nThe gallery below shows the top sampled frames for your query."
    return status, ans, f"Searched for: {', '.join(pipe._query_variants(q.strip()))}", rows, gal, note


def find_query(q):
    if not pipe.idx:
        yield "scan a video first", "", "", [], [], ""
        return
    if not q or not q.strip():
        yield "enter a natural-language query", "", "", [], [], ""
        return
    yield "searching...", "", "", [], [], ""
    yielded = False
    for hits in pipe.search_stream(q.strip(), top_k=cfg.query_top_k):
        seg = pipe.prepare_hits(hits, q.strip())
        status = "matches ready" if seg else "search complete"
        yield _find_payload(status, q.strip(), seg)
        yielded = True
    if not yielded:
        seg = pipe.prepare_hits(pipe.search(q.strip(), top_k=cfg.query_top_k), q.strip())
        yield _find_payload("matches ready" if seg else "search complete", q.strip(), seg)


def get_system_status():
    return pipe.warmup_status()


with gr.Blocks(title="Vision Guard", css=css, theme=theme) as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>Vision Guard</h1>
  <p>Step 1: scan the video. Step 2: write a query and review the matching frames.</p>
</div>
"""
    )
    with gr.Row(elem_classes="app-shell"):
        with gr.Column(scale=1, elem_classes="panel"):
            video = gr.Video(label="cctv video", elem_classes="hidden-empty")
            good = [x for x in _sample_videos() if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            scan_btn = gr.Button("step 1: scan video", variant="primary")
            status = gr.Markdown("ready")
            live = gr.Image(label="live indexing preview", interactive=False, elem_classes="hidden-empty")
            info = gr.Markdown(elem_classes="tight-md")
            query = gr.Textbox(label="query", placeholder="person near gate, white car entering, blue truck, umbrella, backpack", interactive=False)
            searched = gr.Markdown(elem_classes="tight-md")
            find_btn = gr.Button("step 2: find matches", interactive=False)

        with gr.Column(scale=2, elem_classes="panel result-stack"):
            answer = gr.Markdown(elem_classes="tight-md")
            table = gr.Dataframe(headers=["Best Frame At", "Clip Window", "Objects", "Summary"], interactive=False, wrap=True)
            gallery = gr.Gallery(label="matched frames", columns=cfg.gallery_columns, height="auto")
            match_md = gr.Markdown(elem_classes="tight-md")

    scan_btn.click(scan_only, [video], [status, live, info, query, find_btn])
    scan_btn.click(lambda: "", None, searched)
    find_btn.click(find_query, [query], [status, answer, searched, table, gallery, match_md])
    demo.load(fn=get_system_status, inputs=None, outputs=status)


if __name__ == "__main__":
    share = _share_enabled()
    server_name = _server_name()
    if server_name == "127.0.0.1":
        print("Open Vision Guard at http://127.0.0.1:7860")
    demo.launch(server_name=server_name, share=share, show_error=True)
