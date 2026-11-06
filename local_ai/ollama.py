"""A small HTTP client for the Ollama server (default http://127.0.0.1:11434).

Only the endpoints this project needs: /api/version, /api/tags, /api/show, /api/ps,
/api/embed and /api/chat. Errors are turned into three exceptions with a plain fix in
the message, because "connection refused" and "404" are the two things people hit first:

- OllamaNotRunning: nothing is listening. Fix: start it with `ollama serve`.
- ModelNotPulled:   the model is not on this machine. Fix: `ollama pull <model>`.
- OllamaError:      anything else the server said, with its own error text.

Facts this client relies on, checked against Ollama 0.40.14 on 6 November 2026
(details in docs/OLLAMA-FACTS.md):
- /api/chat streams by default. We send "stream": false unless a token callback is given.
- A model loads with a 4096-token context unless "num_ctx" is sent, and a longer prompt is
  cut silently. We always send num_ctx and warn when the prompt nearly fills it.
- "think": true on a model without thinking is a 400 error, so think is only sent when asked.
- Tool-call arguments come back as a JSON object on /api/chat (a string on /v1).
"""
import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import requests

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class OllamaNotRunning(OllamaError):
    pass


class ModelNotPulled(OllamaError):
    def __init__(self, model: str, server_text: str = ""):
        super().__init__(f"Model {model!r} is not on this machine ({server_text or 'not found'}). "
                         f"Fix: run `ollama pull {model}`", 404)
        self.model = model


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class ChatReply:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_seconds: float = 0.0
    load_seconds: float = 0.0
    eval_seconds: float = 0.0
    done_reason: str = ""

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.eval_seconds if self.eval_seconds else 0.0


def strip_think(text: str) -> str:
    """Some thinking models (qwen3:4b on Ollama 0.40.14) keep reasoning inside the content even
    with think false, ending it with </think>. Keep only what comes after the last </think>."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 300.0, retries: int = 2,
                 session: Optional[requests.Session] = None, sleep: Callable[[float], None] = time.sleep):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.sleep = sleep
        self.warnings: List[str] = []

    # --- plumbing -------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[dict] = None, stream: bool = False,
                 model: str = ""):
        url = self.host + path
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(method, url, json=body, timeout=self.timeout, stream=stream)
            except requests.ConnectionError as e:
                if attempt < self.retries and "Connection reset" in str(e):
                    self.sleep(0.5 * (attempt + 1))
                    continue
                raise OllamaNotRunning(f"Ollama is not running at {self.host}. Fix: start it with `ollama serve` "
                                       f"(or open the Ollama app).") from e
            if resp.status_code < 400:
                return resp
            text = _error_text(resp)
            if resp.status_code == 404 and "not found" in text:
                raise ModelNotPulled(model or (body or {}).get("model", "?"), text)
            # 500/502/503 can be a model still loading or a busy runner: worth one more try.
            if resp.status_code in (500, 502, 503) and attempt < self.retries:
                self.sleep(1.0 * (attempt + 1))
                continue
            raise OllamaError(f"Ollama said {resp.status_code}: {text}", resp.status_code)
        raise OllamaError("unreachable")  # pragma: no cover

    # --- read-only info -------------------------------------------------------------------
    def version(self) -> str:
        return self._request("GET", "/api/version").json().get("version", "")

    def tags(self) -> List[dict]:
        return self._request("GET", "/api/tags").json().get("models", [])

    def show(self, model: str) -> dict:
        return self._request("POST", "/api/show", {"model": model}, model=model).json()

    def ps(self) -> List[dict]:
        return self._request("GET", "/api/ps").json().get("models", [])

    def capabilities(self, model: str) -> List[str]:
        return list(self.show(model).get("capabilities") or [])

    # --- work -----------------------------------------------------------------------------
    def embed(self, model: str, inputs: List[str], batch: int = 32) -> List[List[float]]:
        """Embeddings for a list of texts, in order. /api/embed returns them L2-normalised."""
        out: List[List[float]] = []
        for i in range(0, len(inputs), batch):
            part = inputs[i:i + batch]
            data = self._request("POST", "/api/embed", {"model": model, "input": part}, model=model).json()
            vecs = data.get("embeddings") or []
            if len(vecs) != len(part):
                raise OllamaError(f"asked for {len(part)} embeddings, got {len(vecs)}")
            out.extend(vecs)
        return out

    def chat(self, model: str, messages: List[dict], tools: Optional[list] = None, format=None,
             options: Optional[dict] = None, think: Optional[bool] = None, keep_alive=None,
             on_token: Optional[Callable[[str], None]] = None) -> ChatReply:
        body = {"model": model, "messages": messages, "stream": on_token is not None,
                "options": dict(options or {})}
        if tools:
            body["tools"] = tools
        if format is not None:
            body["format"] = format
        if think is not None:
            body["think"] = think
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        started = time.monotonic()
        if on_token is None:
            data = self._request("POST", "/api/chat", body, model=model).json()
            message = data.get("message") or {}
            content = message.get("content") or ""
            thinking = message.get("thinking") or ""
            calls = message.get("tool_calls") or []
        else:
            resp = self._request("POST", "/api/chat", body, stream=True, model=model)
            parts, think_parts, calls, data = [], [], [], {}
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = json.loads(raw)
                if line.get("error"):
                    raise OllamaError(f"Ollama stream error: {line['error']}")
                msg = line.get("message") or {}
                if msg.get("content"):
                    parts.append(msg["content"])
                    on_token(msg["content"])
                if msg.get("thinking"):
                    think_parts.append(msg["thinking"])
                calls.extend(msg.get("tool_calls") or [])
                if line.get("done"):
                    data = line
                    break
            content, thinking = "".join(parts), "".join(think_parts)
        reply = ChatReply(
            content=strip_think(content),
            tool_calls=[_tool_call(c) for c in calls],
            thinking=thinking,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            total_seconds=(data.get("total_duration") or 0) / 1e9 or (time.monotonic() - started),
            load_seconds=(data.get("load_duration") or 0) / 1e9,
            eval_seconds=(data.get("eval_duration") or 0) / 1e9,
            done_reason=data.get("done_reason") or "",
        )
        num_ctx = body["options"].get("num_ctx", 4096)
        if reply.prompt_tokens and reply.prompt_tokens >= 0.9 * num_ctx:
            self.warnings.append(f"prompt used {reply.prompt_tokens} of {num_ctx} context tokens; "
                                 f"older text may have been cut. Raise num_ctx or retrieve fewer chunks.")
        return reply


def _tool_call(raw: dict) -> ToolCall:
    fn = raw.get("function") or {}
    args = fn.get("arguments") or {}
    if isinstance(args, str):  # the /v1 shape, or an older server
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": args}
    return ToolCall(name=fn.get("name", ""), arguments=args if isinstance(args, dict) else {"_raw": args},
                    id=raw.get("id", ""))


def _error_text(resp) -> str:
    try:
        return str(resp.json().get("error") or resp.text)
    except ValueError:
        return resp.text.strip()[:300]
