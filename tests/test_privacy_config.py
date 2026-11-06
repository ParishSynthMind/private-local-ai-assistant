import socket
from types import SimpleNamespace as NS

import pytest
import requests

from local_ai.config import Settings, load_key
from local_ai.llm import OllamaAnswerer
from local_ai.privacy import OfflineViolation, check_local_host, offline_guard
from local_ai.rag import Assistant


def test_guard_blocks_outside_connections_before_sending():
    with offline_guard() as net:
        with pytest.raises(OfflineViolation):
            s = socket.socket()
            try:
                s.connect(("93.184.216.34", 443))
            finally:
                s.close()
    assert net.blocked == ["93.184.216.34:443"]


def test_guard_blocks_dns_for_outside_names():
    with offline_guard() as net:
        with pytest.raises(OfflineViolation):
            socket.getaddrinfo("api.anthropic.com", 443)
    assert net.blocked == ["dns:api.anthropic.com"]


def test_guard_blocks_requests_to_the_internet():
    with offline_guard():
        with pytest.raises((OfflineViolation, requests.ConnectionError)):
            requests.get("https://example.com", timeout=2)


def test_full_local_rag_run_makes_only_local_connections(store, fake):
    from local_ai.ollama import OllamaClient

    fresh = OllamaClient(fake.url)  # a new connection pool, so the connection is made inside the guard
    store.embedder.client = fresh
    with offline_guard() as net:
        ans = Assistant(store, OllamaAnswerer("H", client=fresh)).ask("forklift speed limit")
    assert ans.status == "answered" and net.blocked == [] and net.local >= 1


def test_guard_restores_socket_functions():
    before = (socket.socket.connect, socket.getaddrinfo)
    with offline_guard():
        pass
    assert (socket.socket.connect, socket.getaddrinfo) == before


def test_check_local_host():
    assert check_local_host("http://127.0.0.1:11434") == "127.0.0.1"
    assert check_local_host("localhost:11434") == "localhost"
    assert check_local_host("http://[::1]:11434") == "::1"
    with pytest.raises(OfflineViolation, match="ALLOW_REMOTE_OLLAMA"):
        check_local_host("http://192.168.1.50:11434")
    assert check_local_host("http://192.168.1.50:11434", allow_remote=True) == "192.168.1.50"


def test_settings_defaults_are_local():
    s = Settings.from_env({})
    assert s.provider == "ollama" and s.agent_provider == "ollama" and s.ollama_host == "http://127.0.0.1:11434"


def test_settings_from_env():
    s = Settings.from_env({"PROVIDER": "Anthropic", "OLLAMA_HOST": "0.0.0.0:11434", "TOP_K": "3", "NUM_CTX": "16384"})
    assert s.provider == "anthropic" and s.ollama_host == "http://127.0.0.1:11434"
    assert s.top_k == 3 and s.num_ctx == 16384


@pytest.mark.parametrize("env", [{"TOP_K": "0"}, {"TOP_K": "abc"}, {"NUM_CTX": "100"}])
def test_settings_validation(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_load_key_prefers_env_then_keychain():
    env = {"ANTHROPIC_API_KEY": "from-env"}
    assert load_key("ANTHROPIC_API_KEY", env) == "env"
    env = {}
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return NS(returncode=0, stdout="kc-value\n")

    import local_ai.config as cfg

    old = cfg.sys.platform
    cfg.sys.platform = "darwin"
    try:
        assert load_key("GEMINI_API_KEY", env, run=fake_run) == "keychain"
    finally:
        cfg.sys.platform = old
    assert env["GEMINI_API_KEY"] == "kc-value" and calls[0][:4] == ["security", "find-generic-password", "-s",
                                                                     "gemini-api-key"]


def test_load_key_missing():
    import local_ai.config as cfg

    old = cfg.sys.platform
    cfg.sys.platform = "linux"
    try:
        with pytest.raises(RuntimeError, match="PROVIDER=ollama needs none"):
            load_key("OPENAI_API_KEY", {})
    finally:
        cfg.sys.platform = old
