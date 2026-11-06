"""Render the README images headless with Playwright, from the real run output in examples/local-run/.

    python scripts/render_media.py   (needs: pip install playwright && playwright install chromium)

1. media/local-run.png     check, index (with the flagged passage) and the offline network counts
2. media/comparison.png    the side-by-side table and every question's result per model
3. media/agent-run.png     the agent's tool calls on the local model
4. media/web-ui.png        the web page answering on the local model (needs `local-ai serve` running)
Everything shown is copied from the recorded run; nothing is typed in by hand.
"""
import html
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "examples" / "local-run"
MEDIA = ROOT / "media"
MONO = "'SF Mono', Menlo, Consolas, monospace"
SANS = "-apple-system, Helvetica, Arial, sans-serif"


def colour(line: str) -> str:
    t = html.escape(line)
    if line.startswith("$"):
        return f"<span style='color:#7ee787'>{t}</span>"
    if "FLAGGED" in line or "WRONG" in line or " X  " in line:
        return f"<span style='color:#d29922'>{t}</span>"
    if line.startswith("network:") or "outside connections" in line or "RIGHT" in line:
        return f"<span style='color:#3fb950'>{t}</span>"
    if line.strip().startswith("->"):
        return f"<span style='color:#79c0ff'>{t}</span>"
    if line.startswith("--"):
        return f"<span style='color:#d2a8ff'>{t}</span>"
    return t


def terminal(lines, title):
    body = "\n".join(colour(x) for x in lines)
    return f"""<html><body style="margin:0;background:#0d1117"><div id="wrap" style="display:inline-block;padding:22px 28px;background:#0d1117;max-width:1400px">
<div style="font-family:{SANS};color:#8b949e;font-size:13px;margin-bottom:10px">{html.escape(title)}</div>
<pre style="margin:0;font-family:{MONO};font-size:13px;line-height:1.45;color:#c9d1d9;white-space:pre-wrap">{body}</pre></div></body></html>"""


def comparison(data, env):
    s = data["summary"]
    head = ("<tr><th>Model</th><th>Runs on</th><th>Correct</th><th>Cited right file</th><th>Median</th>"
            "<th>Tokens in / out</th><th>Cost (12 questions)</th></tr>")
    trs = ""
    for r in s:
        where = "this Mac, offline" if r["local"] else f"{r['provider']} cloud"
        cost = "$0" if r["local"] else f"${r['cost_usd']:.4f}"
        trs += (f"<tr><td><b>{html.escape(r['model'])}</b></td><td>{where}</td><td>{r['correct']}/{r['questions']}</td>"
                f"<td>{r['cited']}/{r['questions']}</td><td>{r['median_seconds']:.1f}s</td>"
                f"<td>{r['tokens_in']:,} / {r['tokens_out']:,}</td><td>{cost}</td></tr>")
    models = [r["model"] for r in s]
    qids = sorted({r["qid"] for r in data["rows"]})
    grid = "<tr><th>Question</th>" + "".join(f"<th>{html.escape(m)}</th>" for m in models) + "</tr>"
    for q in qids:
        rows = {r["model"]: r for r in data["rows"] if r["qid"] == q}
        question = next(iter(rows.values()))["question"]
        cells = ""
        for m in models:
            r = rows.get(m)
            ok = r and r["correct"]
            cells += (f"<td style='background:{'#dafbe1' if ok else '#ffebe9'}'>{'right' if ok else 'wrong'} "
                      f"<span style='color:#59636e'>{r['answer_seconds']:.1f}s</span></td>")
        grid += f"<tr><td>{html.escape(question)}</td>{cells}</tr>"
    hw = env["hardware"]
    title = (f"Same 12 questions, same retrieved passages. Recorded {env['date']} on {hw.get('Model Name', '')} "
             f"({hw.get('Chip', '')}, {hw.get('Memory', '')}), Ollama {env['ollama']}")
    return f"""<html><body style="margin:0;background:#fff"><div id="wrap" style="display:inline-block;padding:20px 24px;background:#fff;font-family:{SANS}">
<div style="color:#555;font-size:13px;margin-bottom:10px">{html.escape(title)}</div>
<style>table{{border-collapse:collapse;font-size:13px;margin-bottom:16px}} th{{background:#f6f8fa;text-align:left}}
th,td{{border:1px solid #d1d9e0;padding:6px 10px}}</style>
<table>{head}{trs}</table><table>{grid}</table></div></body></html>"""


def main():
    MEDIA.mkdir(exist_ok=True)
    lines = (RUN / "terminal.txt").read_text().splitlines()
    env = json.loads((RUN / "environment.json").read_text())
    data = json.loads((RUN / "comparison.json").read_text())
    start_cmp = next(i for i, x in enumerate(lines) if x.startswith("$ compare"))
    start_agent = next(i for i, x in enumerate(lines) if x.startswith("$ local-ai agent"))
    v1 = next((i for i, x in enumerate(lines) if x.startswith("-- ollama /v1")), len(lines))
    pages = [
        (terminal(lines[:start_cmp], "Recorded live run: Ollama on this Mac, offline guard on"), "local-run.png"),
        (comparison(data, env), "comparison.png"),
        (terminal(lines[start_agent:v1], "Recorded live run: the agent's tool calls on local models"), "agent-run.png"),
    ]
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=2)
        for doc, out in pages:
            page.set_content(doc)
            page.locator("#wrap").screenshot(path=str(MEDIA / out))
            print("wrote media/" + out)
        if "--web" in sys.argv:
            page = b.new_page(viewport={"width": 900, "height": 300}, device_scale_factor=2)
            page.goto("http://127.0.0.1:8765/")
            questions = ["What should I do when alarm AL-214 comes up?", "Are safety glasses optional on night shift?"]
            for i, q in enumerate(questions, 1):
                page.fill("#q", q)
                page.click("button[type=submit]")
                page.wait_for_function(f"document.querySelectorAll('.meta').length >= {i}", timeout=120000)
            page.screenshot(path=str(MEDIA / "web-ui.png"), full_page=True)
            print("wrote media/web-ui.png")
        b.close()


if __name__ == "__main__":
    main()
