import csv
import json
import os
import zipfile
from datetime import datetime
from jinja2 import Template


class ReportGenerator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def write_csv(self, path, rows):
        cols = ["rank", "score", "start", "end", "duration", "summary", "objects", "tracks", "clip"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i, row in enumerate(rows, 1):
                w.writerow({
                    "rank": i,
                    "score": round(row.get("score", 0.0), 4),
                    "start": round(row.get("start", 0.0), 2),
                    "end": round(row.get("end", 0.0), 2),
                    "duration": round(row.get("end", 0.0) - row.get("start", 0.0), 2),
                    "summary": row.get("summary", ""),
                    "objects": ", ".join(row.get("objects", [])),
                    "tracks": ", ".join(str(x) for x in row.get("tracks", [])),
                    "clip": row.get("clip", ""),
                })
        return path

    def write_html(self, path, data):
        tpl = Template(
            """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Vision Guard</title>
  <style>
    body{font-family:system-ui,sans-serif;background:#f3f5f7;color:#14202b;margin:0}
    .wrap{max-width:1080px;margin:0 auto;padding:28px}
    .hero{background:linear-gradient(135deg,#0f3d5e,#2d718f);color:#fff;padding:24px;border-radius:18px}
    .card{background:#fff;border-radius:16px;padding:18px;margin-top:16px;box-shadow:0 8px 24px rgba(18,38,63,.08)}
    .meta{color:#53606d;font-size:14px}
    .score{font-weight:700;color:#0f3d5e}
    table{width:100%;border-collapse:collapse;margin-top:10px}
    th,td{padding:10px;border-bottom:1px solid #e7ecf0;text-align:left;vertical-align:top}
    th{background:#eef4f8}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Vision Guard</h1>
      <div>query: {{ q }}</div>
      <div>video: {{ video }}</div>
      <div>generated: {{ now }}</div>
    </div>
    <div class="card">
      <h2>Top Matches</h2>
      <table>
        <tr>
          <th>#</th>
          <th>Time</th>
          <th>Score</th>
          <th>Summary</th>
          <th>Objects</th>
          <th>Clip</th>
        </tr>
        {% for x in hits %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ "%.2f"|format(x.start) }}s - {{ "%.2f"|format(x.end) }}s</td>
          <td class="score">{{ "%.4f"|format(x.score) }}</td>
          <td>{{ x.summary }}</td>
          <td>{{ x.objects|join(", ") }}</td>
          <td>{{ x.clip }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
  </div>
</body>
</html>
"""
        )
        html = tpl.render(
            q=data.get("query", ""),
            video=data.get("video", ""),
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hits=data.get("hits", []),
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def write_zip(self, path, files):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for file in files:
                if file and os.path.exists(file):
                    z.write(file, os.path.basename(file))
        return path
