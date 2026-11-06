"""A faithful fake of the Ollama HTTP API, so every test runs with no Ollama and no model.

It is a real HTTP server on 127.0.0.1 (a random port) and copies the behaviour recorded from
Ollama 0.40.14 on 6 November 2026 (docs/OLLAMA-FACTS.md), including the awkward parts:

- /api/chat streams NDJSON unless "stream": false is sent (the default is true).
- Unknown model: 404 {"error": "model 'x' not found"}; /api/show adds ":latest" to a bare name.
- Embeddings from a chat model: 501 "This server does not support embeddings...".
- Chat with an embedding model: 400 "\"x\" does not support chat".
- "think": true on a model without thinking: 400 "\"x\" does not support thinking".
- Tools sent to a model without the tools capability: 400 "\"x\" does not support tools".
- Context: a model loads with num_ctx 4096 unless options.num_ctx is sent; a longer prompt is
  cut from the FRONT without an error (the system prompt and early passages disappear).
- /api/embed returns unit-length vectors; tool-call arguments are objects on /api/chat and JSON
  strings on /v1/chat/completions, which reports finish_reason "tool_calls".
- Durations are nanoseconds; prompt_eval_count / eval_count are token counts.

The "model" is a few deterministic rules, enough to drive retrieval, structured answers and
the agent loop end to end: embeddings are hashed bags of words; a chat with a JSON schema
answers from the passage that shares the most words with the question; a chat with tools calls
the calculator for arithmetic and search_documents otherwise, then answers from the result.
Test hooks: fail_next(), script(), requests (a log of every call).
"""
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 768
VERSION = "0.40.14"

DEFAULT_MODELS = {
    "qwen3:4b-instruct": {"family": "qwen3", "parameter_size": "4.0B", "size": 2497293931,
                          "capabilities": ["completion", "tools", "thinking"], "context_length": 262144},
    "llama3.2:3b": {"family": "llama", "parameter_size": "3.2B", "size": 2019393189,
                    "capabilities": ["completion", "tools"], "context_length": 131072},
    "nomic-embed-text:latest": {"family": "nomic-bert", "parameter_size": "137M", "size": 274302450,
                                "capabilities": ["embedding"], "context_length": 2048},
    "gemma3:1b": {"family": "gemma3", "parameter_size": "1B", "size": 815319791,
                  "capabilities": ["completion"], "context_length": 32768},
}

STOP = set("a an and are as at be by can do does for from how i if in is it of on or our the this to "
           "what when where which who why with you your should must we there".split())


def words(text: str):
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def fake_embedding(text: str):
    text = re.sub(r"^(search_document|search_query):\s*", "", text)
    # A hashed bag of words, plus a shared component so unrelated texts score about 0.3, as they
    # do with real nomic-embed-text (measured 6 Nov 2026), and related ones score well above it.
    vec = [0.0] * DIM
    for w in words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[1 + h % (DIM - 1)] += 1.0 if (h >> 12) % 2 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    vec = [x / norm * 0.835 for x in vec]
    vec[0] = 0.55
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def count_tokens(text: str) -> int:
    return max(1, int(len(re.findall(r"\S+", text)) * 1.3))


class FakeOllama:
    def __init__(self, models=None):
        self.models = {k: dict(v) for k, v in (models or DEFAULT_MODELS).items()}
        self.requests = []          # (method, path, body) for every call
        self._fail = []             # queued (status, body) responses
        self._scripts = []          # queued assistant messages for /api/chat
        self.loaded = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    # --- lifecycle ----------------------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # --- test hooks ---------------------------------------------------------------------
    def fail_next(self, status=500, error="llama runner process has terminated", times=1):
        self._fail.extend([(status, {"error": error})] * times)

    def script(self, *messages):
        """Queue exact assistant messages for the next /api/chat calls, e.g.
        {"content": "", "tool_calls": [{"function": {"name": "today", "arguments": {}}}]}."""
        self._scripts.extend(messages)

    def calls(self, path):
        return [b for m, p, b in self.requests if p == path]

    # --- model lookup -------------------------------------------------------------------
    def resolve(self, name: str):
        if not name:
            return None
        full = name if ":" in name else name + ":latest"
        return full if full in self.models else None

    # --- handlers -----------------------------------------------------------------------
    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, body, content_type="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                if not raw:
                    return {}
                return json.loads(raw)

            def do_GET(self):
                fake.requests.append(("GET", self.path, None))
                if fake._fail:
                    return self._send(*fake._fail.pop(0))
                if self.path == "/api/version":
                    return self._send(200, {"version": VERSION})
                if self.path == "/api/tags":
                    return self._send(200, {"models": [fake._tag(n) for n in fake.models]})
                if self.path == "/api/ps":
                    return self._send(200, {"models": [dict(fake._tag(n), context_length=c)
                                                       for n, c in fake.loaded.items()]})
                if self.path == "/v1/models":
                    return self._send(200, {"object": "list", "data": [
                        {"id": n, "object": "model", "created": 0, "owned_by": "library"} for n in fake.models]})
                if self.path == "/":
                    return self._send(200, b"Ollama is running", "text/plain; charset=utf-8")
                return self._send(404, b"404 page not found", "text/plain")

            def do_POST(self):
                try:
                    body = self._body()
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid character looking for beginning of value"})
                fake.requests.append(("POST", self.path, body))
                if fake._fail:
                    return self._send(*fake._fail.pop(0))
                route = {"/api/show": fake._show, "/api/embed": fake._embed, "/api/chat": fake._chat,
                         "/v1/chat/completions": fake._v1_chat}.get(self.path)
                if route is None:
                    return self._send(404, b"404 page not found", "text/plain")
                status, payload = route(body)
                if isinstance(payload, list):  # NDJSON stream
                    data = b"".join(json.dumps(x).encode() + b"\n" for x in payload)
                    return self._send(status, data, "application/x-ndjson")
                return self._send(status, payload)

        return Handler

    def _tag(self, name):
        m = self.models[name]
        return {"name": name, "model": name, "size": m["size"], "digest": hashlib.sha256(name.encode()).hexdigest(),
                "details": {"format": "gguf", "family": m["family"], "parameter_size": m["parameter_size"],
                            "quantization_level": "Q4_K_M"}, "capabilities": m["capabilities"]}

    def _show(self, body):
        name = body.get("model") or body.get("name") or ""
        full = self.resolve(name)
        if not full:
            return 404, {"error": f"model '{name if ':' in name else name + ':latest'}' not found"}
        m = self.models[full]
        return 200, {"details": self._tag(full)["details"], "capabilities": m["capabilities"],
                     "model_info": {f"{m['family']}.context_length": m["context_length"]}, "parameters": ""}

    def _embed(self, body):
        name = body.get("model", "")
        full = self.resolve(name)
        if not full:
            return 404, {"error": f"model '{name}' not found"}
        if "embedding" not in self.models[full]["capabilities"]:
            return 501, {"error": "This server does not support embeddings. Start it with `--embeddings`"}
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        self.loaded[full] = 2048
        return 200, {"model": name, "embeddings": [fake_embedding(t) for t in inputs],
                     "total_duration": 5_000_000, "load_duration": 1_000_000,
                     "prompt_eval_count": sum(count_tokens(t) for t in inputs)}

    # --- chat -----------------------------------------------------------------------------
    def _check_chat(self, body):
        name = body.get("model", "")
        full = self.resolve(name)
        if not full:
            return None, (404, {"error": f"model '{name}' not found"})
        caps = self.models[full]["capabilities"]
        if "completion" not in caps:
            return None, (400, {"error": f'"{name}" does not support chat'})
        if body.get("think") is True and "thinking" not in caps:
            return None, (400, {"error": f'"{name}" does not support thinking'})
        if body.get("tools") and "tools" not in caps:
            return None, (400, {"error": f'"{name}" does not support tools'})
        return full, None

    def _window(self, messages, num_ctx):
        """Ollama keeps the END of an over-long prompt. Drop text from the front until it fits."""
        flat = [(i, m) for i, m in enumerate(messages)]
        total = sum(count_tokens(m.get("content") or "") for _, m in flat)
        truncated = False
        kept = [dict(m) for m in messages]
        while total > num_ctx and kept:
            first = kept[0]
            toks = count_tokens(first.get("content") or "")
            if total - toks >= num_ctx or len(kept) == 1:
                if len(kept) == 1:
                    words_ = (first.get("content") or "").split()
                    keep_n = max(1, int(num_ctx / 1.3))
                    first["content"] = " ".join(words_[-keep_n:])
                    total = count_tokens(first["content"])
                else:
                    kept.pop(0)
                    total -= toks
            else:
                words_ = (first.get("content") or "").split()
                drop = int((total - num_ctx) / 1.3) + 1
                first["content"] = " ".join(words_[drop:])
                total = sum(count_tokens(m.get("content") or "") for m in kept)
            truncated = True
        return kept, total, truncated

    def _brain(self, messages, tools, fmt):
        if self._scripts:
            return dict(self._scripts.pop(0))
        last = messages[-1] if messages else {"role": "user", "content": ""}
        if fmt is not None:
            return {"content": json.dumps(self._rag_answer(messages, fmt))}
        tool_names = {t["function"]["name"] for t in (tools or [])}
        if tools and last.get("role") == "user":
            q = last.get("content") or ""
            math_expr = re.search(r"(\d[\d.,]*\s*(?:[-+*/x×]|percent of|% of)\s*\d[\d.,]*(?:\s*[-+*/]\s*\d[\d.,]*)*)", q)
            if "calculator" in tool_names and math_expr and "search" not in q.lower():
                expr = math_expr.group(1).replace("x", "*").replace("×", "*").replace("percent of", "% of")
                return self._call("calculator", {"expression": expr})
            if "list_documents" in tool_names and re.search(r"\b(list|which|what) (the )?documents\b", q, re.I):
                return self._call("list_documents", {})
            if "search_documents" in tool_names:
                return self._call("search_documents", {"query": q[:120]})
        if tools and last.get("role") == "tool":
            result = last.get("content") or ""
            if result.startswith("ERROR"):
                return {"content": "I could not complete that: " + result[7:120]}
            first_hit = re.split(r"\n\n\[passage", result)[0]
            text = first_hit.split("\n", 1)[1] if "\n" in first_hit else first_hit
            src = re.search(r"\] ([^,(]+)", first_hit)
            source = f" (source: {src.group(1).strip()})" if src else ""
            return {"content": text.strip()[:300] + source}
        return {"content": "Hello from the fake model."}

    def _call(self, name, args):
        cid = hashlib.md5(f"{name}{json.dumps(args)}{time.time_ns()}".encode()).hexdigest()[:32]
        return {"content": "", "tool_calls": [{"id": cid, "function": {"index": 0, "name": name, "arguments": args}}]}

    def _rag_answer(self, messages, fmt):
        user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        q = re.search(r"<question>\s*(.*?)\s*</question>", user, re.S)
        question = q.group(1) if q else user
        qwords = set(words(question))
        best, best_id, best_overlap = None, None, 0
        for m in re.finditer(r'<document id="(\d+)"[^>]*>\s*(.*?)\s*</document>', user, re.S):
            overlap = len(qwords & set(words(m.group(2))))
            if overlap > best_overlap:
                best, best_id, best_overlap = m.group(2), int(m.group(1)), overlap
        if best is None or best_overlap < 1:
            return {"status": "not_in_documents", "answer": "The documents do not say.", "citations": []}
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", best))
        sentence = max(sentences, key=lambda s: len(qwords & set(words(s))))
        return {"status": "answered", "answer": sentence.strip(), "citations": [best_id]}

    def _chat(self, body):
        full, err = self._check_chat(body)
        if err:
            return err
        num_ctx = int((body.get("options") or {}).get("num_ctx") or 4096)
        self.loaded[full] = num_ctx
        messages, prompt_tokens, _ = self._window(body.get("messages") or [], num_ctx)
        msg = self._brain(messages, body.get("tools"), body.get("format"))
        content = msg.get("content", "")
        out_tokens = max(1, count_tokens(content)) if content else 20
        stats = {"done": True, "done_reason": "stop", "total_duration": 900_000_000, "load_duration": 100_000_000,
                 "prompt_eval_count": prompt_tokens, "prompt_eval_duration": 200_000_000,
                 "eval_count": out_tokens, "eval_duration": 600_000_000}
        created = "2026-11-06T15:51:40.353441Z"
        message = {"role": "assistant", "content": content}
        if msg.get("tool_calls"):
            message["tool_calls"] = msg["tool_calls"]
        if body.get("stream", True) is False:
            return 200, dict({"model": body["model"], "created_at": created, "message": message}, **stats)
        pieces = re.findall(r"\S+\s*", content) or [""]
        lines = [{"model": body["model"], "created_at": created,
                  "message": {"role": "assistant", "content": p}, "done": False} for p in pieces]
        if msg.get("tool_calls"):
            lines = [{"model": body["model"], "created_at": created, "message": message, "done": False}]
        lines.append(dict({"model": body["model"], "created_at": created,
                           "message": {"role": "assistant", "content": ""}}, **stats))
        return 200, lines

    def _v1_chat(self, body):
        native = dict(body, stream=False)
        # /v1 sends tool calls back with JSON-string arguments; convert to the native form first.
        msgs = []
        for m in body.get("messages") or []:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                                 "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                                   for c in m["tool_calls"]]
            msgs.append(m)
        native["messages"] = msgs
        status, data = self._chat(native)
        if status != 200:
            return status, {"error": {"message": data["error"], "type": "api_error", "param": None, "code": None}}
        msg = data["message"]
        out = {"role": "assistant", "content": msg.get("content", "")}
        finish = "stop"
        if msg.get("tool_calls"):
            out["tool_calls"] = [{"id": c.get("id", "call"), "index": i, "type": "function",
                                  "function": {"name": c["function"]["name"],
                                               "arguments": json.dumps(c["function"]["arguments"])}}
                                 for i, c in enumerate(msg["tool_calls"])]
            finish = "tool_calls"
        return 200, {"id": "chatcmpl-476", "object": "chat.completion", "created": 0, "model": body.get("model"),
                     "system_fingerprint": "fp_ollama",
                     "choices": [{"index": 0, "message": out, "finish_reason": finish}],
                     "usage": {"prompt_tokens": data["prompt_eval_count"], "completion_tokens": data["eval_count"],
                               "total_tokens": data["prompt_eval_count"] + data["eval_count"]}}
