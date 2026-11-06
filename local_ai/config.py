"""Settings from environment variables, with defaults that work on a fresh install.

Everything is local by default: PROVIDER=ollama, Ollama on 127.0.0.1, the index in ./data.
Cloud keys are only read when a cloud provider is chosen: from the environment first, then
(on macOS) from the login Keychain, so a key never has to sit in a file.
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

KEYCHAIN_ITEMS = {"ANTHROPIC_API_KEY": "anthropic-api-key", "GEMINI_API_KEY": "gemini-api-key",
                  "OPENAI_API_KEY": "openai-api-key"}


@dataclass
class Settings:
    provider: str = "ollama"
    model: str = ""                      # empty = the provider's default
    agent_provider: str = "ollama"
    agent_model: str = ""
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    embed_model: str = "nomic-embed-text"
    docs_dir: Path = Path("samples/docs")
    index_path: Path = Path("data/index.sqlite")
    company: str = "Hayashi Seiko Works"
    top_k: int = 5
    min_score: float = 0.35
    num_ctx: int = 8192
    chunk_chars: int = 900
    screen_passages: bool = True        # leave passages that address the AI out of search

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        e = os.environ if env is None else env
        s = cls()
        s.provider = e.get("PROVIDER", s.provider).strip().lower()
        s.model = e.get("MODEL", s.model).strip()
        s.agent_provider = e.get("AGENT_PROVIDER", s.agent_provider).strip().lower()
        s.agent_model = e.get("AGENT_MODEL", s.agent_model).strip()
        s.ollama_host = e.get("OLLAMA_HOST", s.ollama_host).strip()
        if s.ollama_host and "://" not in s.ollama_host:
            s.ollama_host = "http://" + s.ollama_host  # Ollama itself accepts "127.0.0.1:11434"
        if s.ollama_host.startswith("http://0.0.0.0"):
            s.ollama_host = s.ollama_host.replace("0.0.0.0", "127.0.0.1", 1)  # a bind address, not a target
        s.allow_remote_ollama = e.get("ALLOW_REMOTE_OLLAMA", "") in ("1", "true", "yes")
        s.embed_model = e.get("EMBED_MODEL", s.embed_model).strip()
        s.docs_dir = Path(e.get("DOCS_DIR", str(s.docs_dir)))
        s.index_path = Path(e.get("INDEX_PATH", str(s.index_path)))
        s.company = e.get("COMPANY_NAME", s.company)
        s.top_k = _int(e, "TOP_K", s.top_k, 1, 12)
        s.min_score = float(e.get("MIN_SCORE", s.min_score))
        s.num_ctx = _int(e, "NUM_CTX", s.num_ctx, 2048, 262144)
        s.chunk_chars = _int(e, "CHUNK_CHARS", s.chunk_chars, 200, 4000)
        s.screen_passages = e.get("SCREEN_PASSAGES", "1") not in ("0", "false", "no")
        return s


def _int(env, name, default, lo, hi) -> int:
    try:
        value = int(env.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def load_key(env_name: str, env=None, run=subprocess.run) -> str:
    """Put a cloud API key into the environment for the SDKs, from the env or the macOS Keychain.
    Returns the source ('env' or 'keychain'); the value is never printed or logged."""
    e = os.environ if env is None else env
    if e.get(env_name):
        return "env"
    item = KEYCHAIN_ITEMS.get(env_name)
    if item and sys.platform == "darwin":
        proc = run(["security", "find-generic-password", "-s", item, "-w"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            e[env_name] = proc.stdout.strip()
            return "keychain"
    raise RuntimeError(f"{env_name} is not set (and no Keychain item {item!r}). Cloud providers need a key; "
                       f"PROVIDER=ollama needs none.")
