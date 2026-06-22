import os
import threading
import warnings
from pathlib import Path

import gradio as gr

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

setup_cache()
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameters have been moved from the Blocks constructor to the launch\\(\\) method in Gradio 6\\.0: theme, css.*")
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
.export-files{gap:14px}
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
        caption = f"{x['query']} | {x.get('peak_ts', x['start']):.2f}s | {prefix}{x['summary']}" if i == 0 else f"{x['label']} | {prefix}{x['summary']}"
        out.append((frame_path, caption))
    return out


def scan_only(video):
    if not video:
        yield "upload a video first", None, "", gr.update(interactive=False), "", []
        return
    yield "starting scan", None, "", gr.update(interactive=False), "", []
    meta = None
    for ev in pipe.index_video_iter(video):
        if ev["kind"] == "preview":
            yield ev["status"], ev["image"], "", gr.update(interactive=False), "", []
        else:
            meta = ev["meta"]
    yield "scan complete", None, _meta(meta), gr.update(interactive=True), "", []


def _find_payload(status, q, seg):
    rows = [[round(x.get("peak_ts", x["start"]), 2), f"{x['start']:.2f}s - {x['end']:.2f}s", ", ".join(x["objects"]), x["summary"]] for x in seg]
    ans = _ans(q.strip(), seg)
    choices = [x["label"] for x in seg]
    gal = _gallery(seg)
    if not seg:
        note = "### matched frames\n\nNo strong frame matches were found for this query."
    elif any(x.get("low_confidence") for x in seg):
        note = "### matched frames\n\nThe gallery below shows the nearest available sampled frames. These results are low confidence, so review them carefully before export."
    else:
        note = "### matched frames\n\nThe gallery below shows the top sampled frames for your query. Select any rows you want to export as clips and reports."
    return status, ans, f"Searched for: {', '.join(pipe._query_variants(q.strip()))}", rows, gr.update(choices=choices, value=choices[:1]), gal, note, q.strip(), seg, gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def find_query(q):
    blank_pick = gr.update(choices=[], value=[])
    if not pipe.idx:
        yield "scan a video first", "", "", [], blank_pick, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
        return
    if not q or not q.strip():
        yield "enter a natural-language query", "", "", [], blank_pick, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
        return
    yield "searching...", "", "", [], blank_pick, [], "", q.strip(), [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    yielded = False
    for hits in pipe.search_stream(q.strip(), top_k=4):
        seg = pipe.prepare_hits(hits, q.strip())
        status = "matches ready" if seg else "search complete"
        yield _find_payload(status, q.strip(), seg)
        yielded = True
    if not yielded:
        seg = pipe.prepare_hits(pipe.search(q.strip(), top_k=4), q.strip())
        yield _find_payload("matches ready" if seg else "search complete", q.strip(), seg)


def export_selected(picks, q, hits):
    if not hits or not picks:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    pipe.last_hits = hits
    zipf, html, csv = pipe.export_selected(picks, q)
    return gr.update(visible=True, value=zipf), gr.update(visible=True, value=html), gr.update(visible=True, value=csv)


def get_system_status():
    return pipe.warmup_status()


with gr.Blocks(title="Vision Guard", css=css, theme=theme) as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>Vision Guard</h1>
  <p>Step 1: scan the video. Step 2: write a query and find matches. Then review each match and export only what you want.</p>
</div>
"""
    )
    q_state = gr.State("")
    hits_state = gr.State([])

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
            pick = gr.CheckboxGroup(label="choose clips to export")
            export_btn = gr.Button("export selected")
            with gr.Row(elem_classes="export-files"):
                zipf = gr.File(label="zip", visible=False)
                html = gr.File(label="html report", visible=False)
                csv = gr.File(label="csv report", visible=False)
            gallery = gr.Gallery(label="matched frames", columns=2, height="auto")
            match_md = gr.Markdown(elem_classes="tight-md")

    scan_btn.click(scan_only, [video], [status, live, info, query, q_state, hits_state])
    scan_btn.click(lambda: gr.update(interactive=True), None, find_btn)
    scan_btn.click(lambda: "", None, searched)
    find_btn.click(find_query, [query], [status, answer, searched, table, pick, gallery, match_md, q_state, hits_state, zipf, html, csv])
    export_btn.click(export_selected, [pick, q_state, hits_state], [zipf, html, csv])
    demo.load(fn=get_system_status, inputs=None, outputs=status)


if __name__ == "__main__":
    share = _share_enabled()
    server_name = _server_name()
    if server_name == "127.0.0.1":
        print("Open Vision Guard at http://127.0.0.1:7860")
    demo.launch(server_name=server_name, share=share, show_error=True)
