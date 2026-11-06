import json

import pytest

from local_ai.ollama import ModelNotPulled, OllamaClient, OllamaError, OllamaNotRunning, strip_think


def test_version_and_tags(client):
    assert client.version() == "0.40.14"
    names = [m["name"] for m in client.tags()]
    assert "qwen3:4b-instruct" in names and "nomic-embed-text:latest" in names


def test_capabilities(client):
    assert "tools" in client.capabilities("qwen3:4b-instruct")
    assert client.capabilities("nomic-embed-text") == ["embedding"]


def test_not_running_gives_the_fix():
    c = OllamaClient("http://127.0.0.1:9", timeout=2, sleep=lambda s: None)
    with pytest.raises(OllamaNotRunning, match="ollama serve"):
        c.version()


def test_missing_model_says_pull(client):
    with pytest.raises(ModelNotPulled, match="ollama pull nope:1b"):
        client.chat("nope:1b", [{"role": "user", "content": "hi"}])


def test_show_missing_model(client):
    with pytest.raises(ModelNotPulled):
        client.show("nope")


def test_embed_returns_unit_vectors_in_order(client):
    vecs = client.embed("nomic-embed-text", ["coolant concentration", "forklift speed"])
    assert len(vecs) == 2 and len(vecs[0]) == 768
    assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-6
    assert vecs[0] != vecs[1]


def test_embed_batches(client, fake):
    client.embed("nomic-embed-text", [f"text {i}" for i in range(70)], batch=32)
    assert [len(b["input"]) for b in fake.calls("/api/embed")] == [32, 32, 6]


def test_embed_with_chat_model_is_501(client):
    with pytest.raises(OllamaError) as e:
        client.embed("qwen3:4b-instruct", ["hi"])
    assert e.value.status == 501 and "does not support embeddings" in str(e.value)


def test_chat_with_embedding_model_is_400(client):
    with pytest.raises(OllamaError, match="does not support chat"):
        client.chat("nomic-embed-text", [{"role": "user", "content": "hi"}])


def test_think_true_on_model_without_thinking_is_400(client):
    with pytest.raises(OllamaError, match="does not support thinking"):
        client.chat("llama3.2:3b", [{"role": "user", "content": "hi"}], think=True)


def test_tools_on_model_without_tools_is_400(client):
    with pytest.raises(OllamaError, match="does not support tools"):
        client.chat("gemma3:1b", [{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}])


def test_chat_sends_stream_false_and_options(client, fake):
    r = client.chat("qwen3:4b-instruct", [{"role": "user", "content": "hello"}], options={"num_ctx": 8192})
    body = fake.calls("/api/chat")[-1]
    assert body["stream"] is False and body["options"]["num_ctx"] == 8192 and "think" not in body
    assert r.content == "Hello from the fake model."
    assert r.prompt_tokens > 0 and r.output_tokens > 0
    assert r.total_seconds == pytest.approx(0.9) and r.eval_seconds == pytest.approx(0.6)
    assert r.tokens_per_second > 0


def test_streaming_collects_tokens(client):
    seen = []
    r = client.chat("qwen3:4b-instruct", [{"role": "user", "content": "hello"}], on_token=seen.append)
    assert "".join(seen) == "Hello from the fake model."
    assert r.content == "Hello from the fake model." and r.output_tokens > 0


def test_tool_call_arguments_are_objects(client, fake):
    fake.script({"content": "", "tool_calls": [{"id": "abc", "function": {"index": 0, "name": "calculator",
                                                                          "arguments": {"expression": "2+2"}}}]})
    r = client.chat("qwen3:4b-instruct", [{"role": "user", "content": "x"}],
                    tools=[{"type": "function", "function": {"name": "calculator", "parameters": {}}}])
    assert r.tool_calls[0].name == "calculator" and r.tool_calls[0].arguments == {"expression": "2+2"}
    assert r.tool_calls[0].id == "abc"


def test_string_arguments_are_parsed_and_garbage_is_kept_raw(client, fake):
    fake.script({"content": "", "tool_calls": [{"function": {"name": "a", "arguments": json.dumps({"k": 1})}}]},
                {"content": "", "tool_calls": [{"function": {"name": "b", "arguments": "{not json"}}]})
    tools = [{"type": "function", "function": {"name": "a", "parameters": {}}}]
    assert client.chat("qwen3:4b-instruct", [{"role": "user", "content": "x"}], tools=tools).tool_calls[0].arguments == {"k": 1}
    assert "_raw" in client.chat("qwen3:4b-instruct", [{"role": "user", "content": "x"}], tools=tools).tool_calls[0].arguments


def test_retries_a_500_then_succeeds(client, fake):
    fake.fail_next(500, times=1)
    assert client.version() == "0.40.14"


def test_gives_up_after_retries(client, fake):
    fake.fail_next(503, "server busy", times=5)
    with pytest.raises(OllamaError, match="503"):
        client.version()


def test_400_is_not_retried(client, fake):
    fake.fail_next(400, "bad request", times=1)
    with pytest.raises(OllamaError, match="400"):
        client.version()
    assert client.version() == "0.40.14"


def test_strip_think():
    assert strip_think("We are asked...\n</think>\n\nHi what's up") == "Hi what's up"
    assert strip_think("  plain answer ") == "plain answer"


def test_warns_when_prompt_nearly_fills_context(client):
    long = "word " * 4000
    client.chat("qwen3:4b-instruct", [{"role": "user", "content": long}], options={"num_ctx": 4096})
    assert client.warnings and "context tokens" in client.warnings[0]
