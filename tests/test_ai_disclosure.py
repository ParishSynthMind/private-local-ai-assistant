"""EU AI Act Art. 50: every conversation opens with a clear AI disclosure, on every user-facing path."""
import json

import pytest
from fastapi.testclient import TestClient

from local_ai import cli
from local_ai.app import build
from local_ai.cli import main
from local_ai.config import DEFAULT_AI_DISCLOSURE, Settings
from local_ai.web import create_app

DEFAULT = "I'm an AI assistant for Hayashi Seiko Works."


@pytest.fixture
def settings(fake, docs, tmp_path):
    s = Settings.from_env({})
    s.ollama_host, s.docs_dir, s.index_path = fake.url, docs, tmp_path / "d.sqlite"
    return s


@pytest.fixture
def app(settings):
    a = build(settings)
    a.store.index_folder(settings.docs_dir)
    return a


def run_cli(argv, settings):
    lines = []
    code = main(argv, settings=settings, out=lines.append)
    return code, lines


def test_default_names_the_company_and_says_ai():
    s = Settings.from_env({})
    assert s.disclosure == DEFAULT and "AI" in s.disclosure
    assert "{company}" in DEFAULT_AI_DISCLOSURE


def test_blank_text_without_written_request_keeps_the_default():
    assert Settings.from_env({"AI_DISCLOSURE": ""}).disclosure == DEFAULT
    assert Settings.from_env({"AI_DISCLOSURE": "   "}).disclosure == DEFAULT
    s = Settings.from_env({"AI_DISCLOSURE": "", "AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST": "  "})
    assert s.disclosure == DEFAULT


def test_custom_text_and_recorded_removal():
    s = Settings.from_env({"AI_DISCLOSURE": "You are chatting with an AI for {company}.", "COMPANY_NAME": "Acme"})
    assert s.disclosure == "You are chatting with an AI for Acme."
    off = Settings.from_env({"AI_DISCLOSURE_REMOVED_BY_WRITTEN_REQUEST": "2026-11-17 email from Acme ops"})
    assert off.disclosure == ""


def test_chat_discloses_once_per_conversation(app):
    answers = iter(["forklift speed limit in aisles", "where is the emergency assembly point?", ""])
    lines = []
    cli.cmd_chat(app, None, out=lines.append, read=lambda p: next(answers))
    assert lines[0].startswith(DEFAULT)                       # first message of the conversation
    assert sum(DEFAULT in x for x in lines) == 1              # not repeated on the second answer
    again = iter(["forklift speed limit", ""])
    lines2 = []
    cli.cmd_chat(app, None, out=lines2.append, read=lambda p: next(again))
    assert lines2[0].startswith(DEFAULT)                      # a new conversation discloses again


def test_chat_blank_disclosure_still_discloses(app):
    app.settings.ai_disclosure = ""
    lines = []
    cli.cmd_chat(app, None, out=lines.append, read=lambda p: "")
    assert lines[0].startswith(DEFAULT)


def test_chat_removed_by_written_request(app):
    app.settings.ai_disclosure_removed_by = "2026-11-17, written request from client"
    lines = []
    cli.cmd_chat(app, None, out=lines.append, read=lambda p: "")
    assert not any("AI assistant for" in x for x in lines)


def test_cli_ask_and_agent_open_with_the_disclosure(settings):
    run_cli(["index"], settings)
    code, lines = run_cli(["ask", "forklift", "speed", "limit"], settings)
    assert code == 0 and lines[0] == DEFAULT and sum(DEFAULT in x for x in lines) == 1
    code, lines = run_cli(["agent", "What is 10 + 20?"], settings)
    assert code == 0 and lines[0] == DEFAULT                  # before the tool steps
    code, lines = run_cli(["ask", "--json", "forklift speed limit"], settings)
    assert json.loads(lines[0])["ai_disclosure"] == DEFAULT


def test_cli_blank_disclosure_without_reference_is_not_dropped(settings):
    settings.ai_disclosure = ""
    run_cli(["index"], settings)
    code, lines = run_cli(["ask", "forklift speed limit"], settings)
    assert lines[0] == DEFAULT


def test_web_page_and_first_api_turn_disclose(app):
    web = TestClient(create_app(app))
    page = web.get("/").text
    assert 'id="ai-disclosure"' in page and DEFAULT.replace("'", "&#x27;") in page
    first = web.post("/api/ask", json={"question": "Where is the emergency assembly point?"}).json()
    assert first["ai_disclosure"] == DEFAULT
    second = web.post("/api/ask", json={"question": "and the forklift speed limit?", "history": [
        {"role": "user", "text": "Where is the emergency assembly point?"},
        {"role": "assistant", "text": first["answer"]}]}).json()
    assert "ai_disclosure" not in second                      # once per conversation
    g = web.post("/api/agent", json={"task": "What is 10 + 20?"}).json()
    assert g["ai_disclosure"] == DEFAULT


def test_web_blank_and_removed(app):
    app.settings.ai_disclosure = ""
    web = TestClient(create_app(app))
    assert 'id="ai-disclosure"' in web.get("/").text
    app.settings.ai_disclosure_removed_by = "2026-11-17, written request from client"
    assert 'id="ai-disclosure"' not in web.get("/").text
    assert "ai_disclosure" not in web.post("/api/ask", json={"question": "forklift speed limit"}).json()
