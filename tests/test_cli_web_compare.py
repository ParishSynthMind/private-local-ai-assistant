import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_ai.app import build
from local_ai.cli import main
from local_ai.compare import Question, load_questions, markdown_table, run_comparison, score
from local_ai.config import Settings
from local_ai.llm import OllamaAnswerer
from local_ai.rag import Answer, Citation
from local_ai.web import create_app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(fake, docs, tmp_path):
    s = Settings.from_env({})
    s.ollama_host, s.docs_dir, s.index_path = fake.url, docs, tmp_path / "cli.sqlite"
    return s


def run_cli(argv, settings):
    lines = []
    code = main(argv, settings=settings, out=lines.append)
    return code, "\n".join(lines)


def test_cli_check(settings):
    code, out = run_cli(["check"], settings)
    assert code == 0 and "Ollama 0.40.14" in out and "qwen3:4b-instruct" in out and "embeddings" in out


def test_cli_check_reports_missing_model(settings):
    settings.model = "mistral:7b"
    code, out = run_cli(["check"], settings)
    assert code == 1 and "ollama pull mistral:7b" in out


def test_cli_check_when_ollama_is_down(settings):
    settings.ollama_host = "http://127.0.0.1:9"
    code, out = run_cli(["check"], settings)
    assert code == 1 and "ollama serve" in out


def test_cli_index_ask_agent(settings):
    code, out = run_cli(["index"], settings)
    assert code == 0 and "Indexed 4 new" in out and "0 outside (blocked)" in out
    code, out = run_cli(["ask", "What", "is", "the", "forklift", "speed", "limit?"], settings)
    assert code == 0 and "[LOCAL, offline]" in out and "Safety-Rules.md" in out and "0 outside" in out
    code, out = run_cli(["ask", "--json", "forklift speed limit"], settings)
    assert json.loads(out)["local"] is True
    code, out = run_cli(["agent", "What is 380 * 0.08?"], settings)
    assert code == 0 and "-> calculator" in out and "30.4" in out


def test_cli_flagged(settings):
    run_cli(["index"], settings)
    code, out = run_cli(["flagged"], settings)
    assert code == 0 and "Shift-Handover-FAQ.md" in out and "ignore your previous instructions" in out


def test_cli_refuses_remote_ollama(settings):
    settings.ollama_host = "http://10.0.0.5:11434"
    code, out = run_cli(["check"], settings)
    assert code == 2 and "not this machine" in out


def test_cli_missing_docs_folder(settings, tmp_path):
    settings.docs_dir = tmp_path / "nope"
    assert run_cli(["index"], settings)[0] == 1


def test_cli_chat_loop(settings, monkeypatch):
    from local_ai import cli

    app = build(settings)
    app.store.index_folder(settings.docs_dir)
    answers = iter(["forklift speed limit in aisles", ""])
    lines = []
    assert cli.cmd_chat(app, None, out=lines.append, read=lambda p: next(answers)) == 0
    assert any("5 km/h" in x for x in lines)


@pytest.fixture
def web(settings):
    app = build(settings)
    app.store.index_folder(settings.docs_dir)
    return TestClient(create_app(app))


def test_web_page_and_health(web):
    r = web.get("/")
    assert r.status_code == 200 and "on this machine, offline" in r.text and "qwen3:4b-instruct" in r.text
    h = web.get("/api/health").json()
    assert h["ok"] and h["local"] and h["passages"] > 0


def test_web_ask_and_agent(web):
    a = web.post("/api/ask", json={"question": "Where is the emergency assembly point?"}).json()
    assert a["status"] == "answered" and a["citations"][0]["source"] == "Safety-Rules.md"
    g = web.post("/api/agent", json={"task": "What is 10 + 20?"}).json()
    assert g["steps"][0]["tool"] == "calculator" and g["stopped"] == "answered"


def test_web_validates_input(web):
    assert web.post("/api/ask", json={"question": ""}).status_code == 422
    assert web.post("/api/ask", json={"question": "x" * 3000}).status_code == 422


def test_web_refuses_other_hosts(web):
    r = web.get("/api/health", headers={"host": "evil.example.com"})
    assert r.status_code == 403


def test_questions_file_loads():
    qs = load_questions(ROOT / "samples" / "questions.json")
    assert len(qs) == 12 and sum(not q.answerable for q in qs) == 2


def _ans(status, text, sources=()):
    return Answer("q", status, text, [Citation(1, s, 0, "", s, "") for s in sources])


def test_scoring_rules():
    q = Question("x", "?", [["10 minutes"], ["30%", "30 percent"]], ["m.pdf"])
    assert score(q, _ans("answered", "Run it 10 minutes at 30 percent.", ["m.pdf"])) == \
        {"correct": True, "cited": True, "forbidden_hit": []}
    assert score(q, _ans("answered", "Run it 10 minutes.", ["other.md"]))["correct"] is False
    out = Question("o", "?", answerable=False)
    assert score(out, _ans("not_in_documents", "no"))["correct"] is True
    assert score(out, _ans("answered", "made up"))["correct"] is False
    bad = Question("b", "?", [["glasses"]], ["s.md"], forbidden=["are optional"])
    assert score(bad, _ans("answered", "Safety glasses are optional.", ["s.md"]))["correct"] is False


def test_comparison_end_to_end_on_fake(store, client):
    qs = load_questions(ROOT / "samples" / "questions.json")
    result = run_comparison(store, [OllamaAnswerer("H", client=client)], qs)
    assert len(result["rows"]) == 12
    s = result["summary"][0]
    assert s["local"] and s["cost_usd"] == 0 and s["questions"] == 12 and s["errors"] == 0
    assert "| qwen3:4b-instruct | this Mac, offline |" in markdown_table(result["summary"])


def test_comparison_survives_a_failing_provider(store):
    class Broken:
        provider, model, local = "anthropic", "claude-sonnet-5-5", False

        def answer(self, *a):
            raise RuntimeError("401 invalid key")

    result = run_comparison(store, [Broken()], [Question("q", "coolant concentration", [["6%"]], ["x"])])
    assert result["rows"][0]["status"] == "error" and result["summary"][0]["errors"] == 1
