"""Command line.

    local-ai check                      is Ollama running, are the models pulled, what will run where
    local-ai index [--rebuild]          read DOCS_DIR into the local index (only changed files)
    local-ai ask "question"             answer from the documents, with sources
    local-ai agent "task"               tool-using agent (search, read, calculator, date)
    local-ai chat                       ask questions in a loop (keeps the conversation)
    local-ai flagged                    passages left out of search because they address the AI
    local-ai serve [--port 8765]        small web page on 127.0.0.1

Local commands run inside the offline guard: any connection that is not to this machine is
blocked and reported. Add --json for machine-readable output.
"""
import argparse
import contextlib
import json
import platform
import sys

from .app import build
from .config import Settings
from .ollama import ModelNotPulled, OllamaError, OllamaNotRunning
from .privacy import OfflineViolation, offline_guard
from .store import IndexMismatch


def _guard(local: bool):
    return offline_guard() if local else contextlib.nullcontext()


def print_answer(ans, out=print):
    badge = "LOCAL, offline" if ans.local else f"CLOUD ({ans.provider})"
    out(f"[{badge}] {ans.model}")
    out(ans.text)
    if ans.citations:
        out("Sources:")
        for c in ans.citations:
            out(f"  [{c.id}] {c.label}")
    if ans.status == "answered" and not ans.grounded:
        out("  (no valid source was cited: check this answer before relying on it)")
    out(f"retrieval {ans.retrieval_seconds:.2f}s, answer {ans.answer_seconds:.1f}s, "
        f"{ans.input_tokens} tokens in, {ans.output_tokens} out")


def cmd_check(app, args, out=print) -> int:
    s, ok = app.settings, True
    try:
        out(f"Ollama {app.ollama.version()} at {s.ollama_host}")
    except OllamaNotRunning as e:
        out(str(e))
        return 1
    have = {m["name"]: m for m in app.ollama.tags()}
    needed = [(s.embed_model, "embeddings")]
    if s.provider == "ollama":
        needed.append((app.assistant.answerer.model, "answers"))
    if s.agent_provider == "ollama":
        needed.append((app.agent.backend.model, "agent"))
    for name, role in needed:
        full = name if ":" in name else name + ":latest"
        if full not in have:
            out(f"  MISSING {name} ({role}): run `ollama pull {name}`")
            ok = False
            continue
        caps = app.ollama.capabilities(name)
        need = {"embeddings": "embedding", "answers": "completion", "agent": "tools"}[role]
        size = have[full].get("size", 0) / 1e9
        flag = "ok" if need in caps else f"WRONG MODEL: no '{need}' capability"
        ok = ok and need in caps
        out(f"  {name:24} {size:4.1f} GB  {role:10} {flag}")
    out(f"Answers: {s.provider}{'' if s.provider == 'ollama' else ' (CLOUD: question + retrieved passages are sent)'}")
    out(f"Agent:   {s.agent_provider}")
    out(f"Index:   {s.index_path} ({app.store.count()} passages)")
    out(f"Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    return 0 if ok else 1


def cmd_index(app, args, out=print) -> int:
    s = app.settings
    if not s.docs_dir.is_dir():
        out(f"DOCS_DIR {s.docs_dir} is not a folder")
        return 1
    rep = app.store.index_folder(s.docs_dir, rebuild=args.rebuild, max_chars=s.chunk_chars, progress=out)
    out(f"Indexed {len(rep.added)} new, {len(rep.updated)} changed, {len(rep.unchanged)} unchanged, "
        f"{len(rep.removed)} removed, {len(rep.failed)} failed: {rep.chunks} passages in {rep.seconds:.1f}s")
    if rep.flagged:
        out(f"{len(rep.flagged)} passage(s) flagged and left out of search: see `local-ai flagged`")
    return 1 if rep.failed else 0


def cmd_flagged(app, args, out=print) -> int:
    rows = app.store.flagged()
    if not rows:
        out("No passages are flagged.")
    for r in rows:
        where = f"page {r['page']}" if r["page"] else r["heading"]
        out(f"[{r['id']}] {r['source']}, {where}\n    matched: {r['reason']!r}\n    {' '.join(r['text'].split())[:200]}")
    return 0


def cmd_ask(app, args, out=print) -> int:
    ans = app.assistant.ask(" ".join(args.question))
    if args.json:
        out(json.dumps(ans.to_dict(), indent=2))
    else:
        print_answer(ans, out)
    for w in app.ollama.warnings:
        out(f"warning: {w}")
    return 0


def cmd_agent(app, args, out=print) -> int:
    run = app.agent.run(" ".join(args.task), on_step=None if args.json else
                        (lambda st: out(f"  -> {st.tool}({json.dumps(st.arguments)}) "
                                        f"{st.result.splitlines()[0][:90] if st.result else ''}")))
    if args.json:
        out(json.dumps(run.to_dict(), indent=2))
    else:
        out(run.answer)
        out(f"[{run.provider} {run.model}] {len(run.steps)} tool calls, {run.model_calls} model calls, "
            f"{run.seconds:.1f}s, stopped: {run.stopped}")
    return 0 if run.stopped == "answered" else 1


def cmd_chat(app, args, out=print, read=input) -> int:
    history = []
    out("Ask about the documents. Empty line to quit.")
    while True:
        try:
            q = read("> ").strip()
        except EOFError:
            break
        if not q:
            break
        ans = app.assistant.ask(q, history=history[-6:])
        print_answer(ans, out)
        history += [{"role": "user", "text": q}, {"role": "assistant", "text": ans.text}]
    return 0


def cmd_serve(app, args, out=print) -> int:
    import uvicorn

    from .web import create_app

    out(f"Open http://127.0.0.1:{args.port}  (bound to this machine only)")
    uvicorn.run(create_app(app), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="local-ai", description="Private AI assistant on a local model (Ollama).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pi = sub.add_parser("index")
    pi.add_argument("--rebuild", action="store_true")
    pa = sub.add_parser("ask")
    pa.add_argument("question", nargs="+")
    pa.add_argument("--json", action="store_true")
    pg = sub.add_parser("agent")
    pg.add_argument("task", nargs="+")
    pg.add_argument("--json", action="store_true")
    sub.add_parser("chat")
    sub.add_parser("flagged")
    ps = sub.add_parser("serve")
    ps.add_argument("--port", type=int, default=8765)
    return p


COMMANDS = {"check": cmd_check, "index": cmd_index, "ask": cmd_ask, "agent": cmd_agent, "chat": cmd_chat,
            "serve": cmd_serve, "flagged": cmd_flagged}


def main(argv=None, settings: Settings = None, out=print, **build_kw) -> int:
    args = parser().parse_args(argv)
    try:
        settings = settings or Settings.from_env()
        app = build(settings, **build_kw)
        local = {"ask": settings.provider == "ollama", "agent": settings.agent_provider == "ollama",
                 "chat": settings.provider == "ollama"}.get(args.cmd, True)
        if args.cmd == "serve":
            local = False  # uvicorn manages its own sockets; it is bound to 127.0.0.1 instead
        with _guard(local) as net:
            code = COMMANDS[args.cmd](app, args, out=out)
        if net is not None and args.cmd != "check" and not getattr(args, "json", False):
            out(f"network: {net.local} local connections, {len(net.blocked)} outside (blocked)")
        return code
    except (OllamaNotRunning, ModelNotPulled, IndexMismatch, OfflineViolation) as e:
        out(f"error: {e}")
        return 2
    except (OllamaError, ValueError, RuntimeError) as e:
        out(f"error: {e}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
