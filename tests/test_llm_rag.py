import json
from types import SimpleNamespace as NS

import pytest

from local_ai.llm import (
    PROVIDERS,
    ClaudeAnswerer,
    GeminiAnswerer,
    OllamaAnswerer,
    OpenAIAnswerer,
    answer_schema,
    build_user_message,
    cost_usd,
    make_answerer,
    parse_answer,
)
from local_ai.rag import NOTHING_FOUND, Assistant

GOOD = json.dumps({"status": "answered", "answer": "Between 6% and 8%.", "citations": [1]})


def test_ollama_answer_end_to_end(store, client, fake):
    a = Assistant(store, OllamaAnswerer("Hayashi", client=client))
    ans = a.ask("What coolant concentration do we need to keep?")
    assert ans.status == "answered" and ans.grounded and ans.local
    assert "6%" in ans.text and ans.sources == ["HX-500-Maintenance-Manual.pdf"]
    body = fake.calls("/api/chat")[-1]
    assert body["format"] == answer_schema(5) and body["options"]["num_ctx"] == 8192
    assert body["options"]["temperature"] == 0.0 and body["stream"] is False


def test_out_of_scope_question_is_refused(store, client):
    a = Assistant(store, OllamaAnswerer("Hayashi", client=client))
    ans = a.ask("What was the revenue last year?")
    assert ans.status == "not_in_documents" and ans.citations == []


def test_no_hits_means_no_model_call(store, client, fake):
    a = Assistant(store, OllamaAnswerer("Hayashi", client=client), min_score=0.99)
    before = len(fake.calls("/api/chat"))
    ans = a.ask("coolant")
    assert ans.text == NOTHING_FOUND and len(fake.calls("/api/chat")) == before


def test_invalid_citations_are_dropped_and_flagged(store, client, fake):
    fake.script({"content": json.dumps({"status": "answered", "answer": "Made up.", "citations": [9]})})
    ans = Assistant(store, OllamaAnswerer("Hayashi", client=client)).ask("coolant concentration")
    assert ans.citations == [] and ans.dropped_citations == [9] and ans.grounded is False


def test_duplicate_citations_collapse(store, client, fake):
    fake.script({"content": json.dumps({"status": "answered", "answer": "x", "citations": [1, 1]})})
    ans = Assistant(store, OllamaAnswerer("Hayashi", client=client)).ask("coolant concentration")
    assert [c.id for c in ans.citations] == [1]


def test_refusal_carries_no_sources(store, client, fake):
    fake.script({"content": json.dumps({"status": "not_in_documents", "answer": "Not there.", "citations": [1]})})
    ans = Assistant(store, OllamaAnswerer("Hayashi", client=client)).ask("coolant concentration")
    assert ans.citations == [] and ans.grounded


def test_cut_off_local_reply_raises(store, client, fake):
    fake.script({"content": "{"})
    with pytest.raises(Exception):
        Assistant(store, OllamaAnswerer("Hayashi", client=client)).ask("coolant concentration")


def test_empty_question_rejected(store, client):
    with pytest.raises(ValueError):
        Assistant(store, OllamaAnswerer("Hayashi", client=client)).ask("  ")


def test_parse_answer_tolerates_fences_and_preamble():
    assert parse_answer("```json\n" + GOOD + "\n```")["status"] == "answered"
    assert parse_answer("Here you go: " + GOOD)["citations"] == [1]
    with pytest.raises(json.JSONDecodeError):
        parse_answer("no json here")


def test_user_message_marks_documents_as_data(store):
    hits = store.search("coolant", k=2)
    msg = build_user_message("q?", [{"role": "user", "text": "earlier"}], hits)
    assert '<document id="1" source="HX-500-Maintenance-Manual.pdf, page 2">' in msg
    assert "<question>\nq?\n</question>" in msg and "user: earlier" in msg


def test_claude_answerer_request_shape(store):
    sent = {}

    def create(**kw):
        sent.update(kw)
        return NS(stop_reason="end_turn", usage=NS(input_tokens=900, output_tokens=40),
                  content=[NS(type="text", text=GOOD)])

    a = ClaudeAnswerer("Hayashi", client=NS(messages=NS(create=create)))
    ans = Assistant(store, a).ask("coolant concentration")
    assert sent["model"] == "claude-sonnet-5-5" and sent["output_config"]["format"]["type"] == "json_schema"
    assert ans.status == "answered" and not ans.local and ans.input_tokens == 900


def test_claude_refusal(store):
    resp = NS(stop_reason="refusal", usage=NS(input_tokens=1, output_tokens=1), content=[])
    a = ClaudeAnswerer("Hayashi", client=NS(messages=NS(create=lambda **k: resp)))
    assert Assistant(store, a).ask("coolant").status == "not_in_documents"


def test_gemini_answerer(store):
    sent = {}

    def gen(**kw):
        sent.update(kw)
        return NS(text=GOOD, usage_metadata=NS(prompt_token_count=800, candidates_token_count=30, thoughts_token_count=5))

    a = GeminiAnswerer("Hayashi", client=NS(models=NS(generate_content=gen)))
    ans = Assistant(store, a).ask("coolant concentration")
    assert sent["config"]["response_json_schema"] == answer_schema(5) and ans.output_tokens == 35


def test_openai_answerer_and_length_cutoff(store):
    def make(finish):
        msg = NS(content=GOOD)
        return NS(choices=[NS(message=msg, finish_reason=finish)], usage=NS(prompt_tokens=5, completion_tokens=6))

    ok = OpenAIAnswerer("H", client=NS(chat=NS(completions=NS(create=lambda **k: make("stop")))))
    assert Assistant(store, ok).ask("coolant").status == "answered"
    cut = OpenAIAnswerer("H", client=NS(chat=NS(completions=NS(create=lambda **k: make("length")))))
    with pytest.raises(RuntimeError, match="cut off"):
        Assistant(store, cut).ask("coolant")


def test_make_answerer_and_unknown_provider(client):
    a = make_answerer("ollama", "H", client=client)
    assert isinstance(a, OllamaAnswerer) and a.model == "qwen3:4b-instruct"
    with pytest.raises(ValueError, match="PROVIDER must be one of"):
        make_answerer("mystery", "H")


def test_costs():
    assert cost_usd("ollama", 10_000, 10_000) == 0.0
    assert cost_usd("anthropic", 1_000_000, 100_000) == pytest.approx(3.0)
    assert set(PROVIDERS) == {"ollama", "anthropic", "gemini", "openai"}


def test_same_interface_for_every_provider():
    for cls in (OllamaAnswerer, ClaudeAnswerer, GeminiAnswerer, OpenAIAnswerer):
        assert callable(getattr(cls, "answer")) and cls.provider in PROVIDERS
