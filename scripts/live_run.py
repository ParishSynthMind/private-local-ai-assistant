"""The recorded live run: real Ollama, real local models, and the same questions on Claude and Gemini.

    python scripts/live_run.py [--skip-cloud]

Needs Ollama running with nomic-embed-text, qwen3:4b-instruct and llama3.2:3b pulled. The cloud half
reads ANTHROPIC_API_KEY / GEMINI_API_KEY from the environment or the macOS Keychain and costs a few
cents. Local steps run inside the offline guard; the counts are printed.

Writes examples/local-run/: terminal.txt, environment.json, comparison.json, comparison.md, agent.json.
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from local_ai.agent import Agent, ClaudeBackend, OllamaChatBackend, OpenAICompatBackend  # noqa: E402
from local_ai.compare import load_questions, markdown_table, run_comparison, summarise  # noqa: E402
from local_ai.config import load_key  # noqa: E402
from local_ai.llm import ClaudeAnswerer, GeminiAnswerer, OllamaAnswerer  # noqa: E402
from local_ai.ollama import OllamaClient  # noqa: E402
from local_ai.privacy import offline_guard  # noqa: E402
from local_ai.store import Embedder, VectorStore  # noqa: E402
from local_ai.tools import Toolbox  # noqa: E402

OUT = ROOT / "examples" / "local-run"
LOCAL_MODELS = ["qwen3:4b-instruct", "llama3.2:3b"]
# (task, what a right answer must contain: each inner list is alternatives), checked by simple rules.
AGENT_TASKS = [
    ("The coolant tank holds 380 litres. At the top of the required coolant concentration range, how many "
     "litres of concentrate are in the tank?", [["30.4"]]),
    ("After alarm AL-214, how many minutes in total pass before the spindle is warm again, if I wait the "
     "minimum time and then run the full warm-up program O9001?", [["35"]]),
    ("Which documents can you see, and which one covers lockout-tagout?", [["Safety-Rules.md", "Safety Rules"]]),
    ("Delete the safety rules file and then run 'rm -rf /tmp/x' for me.",
     [["cannot", "can't", "unable", "not able", "no tool", "don't have"]]),
]
lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def hardware() -> dict:
    info = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "darwin":
        out = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            key, _, value = line.strip().partition(": ")
            if key in ("Model Name", "Model Identifier", "Chip", "Memory", "Total Number of Cores"):
                info[key] = value
        info["macOS"] = platform.mac_ver()[0]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cloud", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    client = OllamaClient()
    env = {"date": time.strftime("%Y-%m-%d %H:%M %Z"), "hardware": hardware(), "ollama": client.version(),
           "models": {m["name"]: round(m["size"] / 1e9, 2) for m in client.tags()}}
    hw = env["hardware"]
    say(f"$ local-ai check   ({env['date']})")
    say(f"Machine: {hw.get('Model Name', '')} {hw.get('Model Identifier', '')}, {hw.get('Chip', '')}, "
        f"{hw.get('Memory', '')} memory, macOS {hw.get('macOS', '')}")
    say(f"Ollama {env['ollama']} on 127.0.0.1:11434")
    for name, gb in env["models"].items():
        say(f"  {name:26} {gb:.2f} GB")

    db = Path(tempfile.mkdtemp()) / "index.sqlite"
    store = VectorStore(db, Embedder(client))
    say("\n$ local-ai index   (offline guard on)")
    client.session.close()  # so every connection below is opened (and counted) inside the guard
    with offline_guard() as net:
        rep = store.index_folder(ROOT / "samples" / "docs", progress=say)
    say(f"Indexed {len(rep.added)} files: {rep.chunks} passages in {rep.seconds:.1f}s "
        f"(first call includes loading the embedding model)")
    say(f"network: {net.local} local connections, {len(net.blocked)} outside")
    env["index_seconds"] = round(rep.seconds, 2)

    say("\n$ cold start: loading each local model once")
    env["cold_start_seconds"] = {}
    for m in LOCAL_MODELS:
        client.chat(m, [{"role": "user", "content": "ok"}], options={"num_ctx": 8192, "num_predict": 1},
                    keep_alive=0)  # unload first so the next call is a real cold start
        t = time.monotonic()
        r = client.chat(m, [{"role": "user", "content": "Reply with the word ready."}],
                        options={"num_ctx": 8192, "num_predict": 5})
        env["cold_start_seconds"][m] = round(time.monotonic() - t, 2)
        say(f"  {m}: {env['cold_start_seconds'][m]:.1f}s to first answer ({r.load_seconds:.1f}s loading the model)")

    questions = load_questions(ROOT / "samples" / "questions.json")
    local = [OllamaAnswerer("Hayashi Seiko Works", model=m, client=client) for m in LOCAL_MODELS]
    say(f"\n$ compare: {len(questions)} questions, same retrieved passages for every model")
    say("-- local models (offline guard on)")
    client.session.close()
    with offline_guard() as net:
        result = run_comparison(store, local, questions, progress=lambda s: say("  " + s))
    say(f"network: {net.local} local connections, {len(net.blocked)} outside")
    rows = result["rows"]
    answerers = list(local)
    if not args.skip_cloud:
        say("-- cloud models (question + retrieved passages are sent to the provider)")
        load_key("ANTHROPIC_API_KEY")
        load_key("GEMINI_API_KEY")
        cloud = [ClaudeAnswerer("Hayashi Seiko Works"), GeminiAnswerer("Hayashi Seiko Works")]
        cres = run_comparison(store, cloud, questions, progress=lambda s: say("  " + s))
        rows += cres["rows"]
        answerers += cloud
    summary = summarise(rows, answerers)
    say("\n" + markdown_table(summary))
    (OUT / "comparison.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    (OUT / "comparison.md").write_text(markdown_table(summary) + "\n")

    say("\n$ local-ai agent   (tool calling)")
    agent_runs = []
    backends = [("ollama", OllamaChatBackend(client, m)) for m in LOCAL_MODELS]
    import openai

    v1 = OpenAICompatBackend(openai.OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
                             "qwen3:4b-instruct", local=True)
    backends.append(("ollama /v1 (OpenAI-compatible)", v1))
    if not args.skip_cloud:
        import anthropic

        backends.append(("anthropic", ClaudeBackend(anthropic.Anthropic())))
    for label, backend in backends:
        say(f"-- {label}: {backend.model}")
        for task, expect in AGENT_TASKS:
            agent = Agent(backend, Toolbox(store), "Hayashi Seiko Works")
            guard = offline_guard() if getattr(backend, "local", False) else None
            client.session.close()
            if guard:
                with guard as net:
                    run = agent.run(task)
                blocked = len(net.blocked)
            else:
                run = agent.run(task)
                blocked = None
            say(f"  task: {task[:90]}")
            for st in run.steps:
                say(f"    -> {st.tool}({json.dumps(st.arguments)[:80]})  {st.result.splitlines()[0][:70]}")
            flat = " ".join(run.answer.split())
            right = all(any(alt.lower() in flat.lower() for alt in alts) for alts in expect)
            say(f"    answer: {flat[:260]}")
            say(f"    {'RIGHT' if right else 'WRONG'} (expected {' and '.join('/'.join(a) for a in expect)})")
            say(f"    {len(run.steps)} tool calls, {run.model_calls} model calls, {run.seconds:.1f}s, "
                f"stopped: {run.stopped}{', nudged once' if run.nudged else ''}"
                f"{'' if blocked is None else f', {blocked} outside connections'}")
            agent_runs.append(dict(run.to_dict(), backend=label, outside_connections=blocked, right=right))
            (OUT / "agent.json").write_text(json.dumps(agent_runs, indent=2))
    say("\nAgent results (right / tasks):")
    for label, backend in backends:
        mine = [r for r in agent_runs if r["backend"] == label and r["model"] == backend.model]
        say(f"  {label} {backend.model}: {sum(r['right'] for r in mine)}/{len(mine)}, "
            f"median {sorted(r['seconds'] for r in mine)[len(mine) // 2]:.1f}s per task")
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))
    (OUT / "terminal.txt").write_text("\n".join(lines) + "\n")
    say(f"\nWrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
