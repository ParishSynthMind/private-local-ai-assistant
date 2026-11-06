import datetime as dt
from types import SimpleNamespace as NS

import pytest

from local_ai.agent import (
    NUDGE,
    Agent,
    BackendReply,
    ClaudeBackend,
    OllamaChatBackend,
    make_backend,
)
from local_ai.ollama import ToolCall
from local_ai.tools import Toolbox, ToolError, calculate, tool_specs


# --- calculator ---------------------------------------------------------------------------------
@pytest.mark.parametrize("expr,expected", [
    ("380*0.08", "30.4"), ("17.5% of 2340", "409.5"), ("10 + 20", "30"), ("2**10", "1024"),
    ("round(10/3, 2)", "3.33"), ("1,200 * 2", "2400"), ("sqrt(16)", "4"), ("50%", "0.5"), ("12 × 7", "84"),
])
def test_calculator(expr, expected):
    assert calculate(expr).endswith("= " + expected)


def test_ambiguous_percent_is_refused():
    # qwen3:4b-instruct really wrote this for "17.5% of 2340" in the first live probe.
    with pytest.raises(ToolError, match="ambiguous"):
        calculate("17.5 % 2340")


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "a + 1", "[1,2][0]",
                                  "lambda: 1", "9**9999", "", "1/0", "x" * 300, "True + 1"])
def test_calculator_refuses_anything_else(expr):
    with pytest.raises(ToolError):
        calculate(expr)


# --- toolbox ------------------------------------------------------------------------------------
def test_tool_specs_shape():
    names = [t["function"]["name"] for t in tool_specs()]
    assert names == ["list_documents", "search_documents", "read_passage", "calculator", "today"]
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tool_specs())


def test_toolbox_runs_read_only_tools(store):
    tb = Toolbox(store, today=lambda: dt.date(2026, 11, 6))
    assert "Safety-Rules.md (4 passages)" in tb.run("list_documents", {})
    found = tb.run("search_documents", {"query": "forklift speed", "k": 2})
    assert "[passage" in found and "5 km/h" in found
    pid = int(found.split("[passage ")[1].split("]")[0])
    assert "km/h" in tb.run("read_passage", {"passage_id": pid})
    assert tb.run("today", {}) == "2026-11-06"
    assert "Safety-Rules.md" in tb.sources_seen


def test_toolbox_errors_are_returned_not_raised(store):
    tb = Toolbox(store)
    assert tb.run("run_shell", {"cmd": "rm -rf /"}).startswith("ERROR: there is no tool called 'run_shell'")
    assert tb.run("read_passage", {"passage_id": 99999}) == "ERROR: no passage 99999"
    assert tb.run("read_passage", {"passage_id": "abc"}).startswith("ERROR")
    assert tb.run("search_documents", {}).startswith("ERROR: query is required")
    assert tb.run("calculator", {"_raw": "{bad"}).startswith("ERROR: the arguments must be a JSON object")


def test_search_k_is_clamped(store):
    out = Toolbox(store).run("search_documents", {"query": "safety", "k": 500})
    assert out.count("[passage") <= 8


# --- the loop, against the fake Ollama --------------------------------------------------------------
def test_agent_calculates_with_the_tool(store, client, fake):
    agent = Agent(OllamaChatBackend(client), Toolbox(store))
    run = agent.run("What is 380 * 0.08?")
    assert run.stopped == "answered" and run.steps[0].tool == "calculator"
    assert "30.4" in run.steps[0].result and run.model_calls == 2
    body = fake.calls("/api/chat")[0]
    assert len(body["tools"]) == 5 and body["messages"][0]["role"] == "system"


def test_agent_searches_documents_and_cites(store, client):
    run = Agent(OllamaChatBackend(client), Toolbox(store)).run("Where do we assemble when the fire alarm sounds?")
    assert run.steps[0].tool == "search_documents" and "gate 3" in run.answer
    assert "Safety-Rules.md" in run.sources


def test_tool_results_go_back_with_tool_name(store, client, fake):
    Agent(OllamaChatBackend(client), Toolbox(store)).run("What is 2 * 3?")
    second = fake.calls("/api/chat")[1]["messages"]
    assert second[-1] == {"role": "tool", "content": "2 * 3 = 6", "tool_name": "calculator"}
    assert second[-2]["tool_calls"][0]["function"] == {"name": "calculator", "arguments": {"expression": "2 * 3"}}


def test_nudge_once_when_model_skips_tools(store, client, fake):
    fake.script({"content": "I don't have that information."})
    run = Agent(OllamaChatBackend(client), Toolbox(store)).run("forklift speed limit in aisles")
    assert run.nudged and run.steps and run.steps[0].tool == "search_documents"
    msgs = fake.calls("/api/chat")[1]["messages"]
    assert msgs[-1] == {"role": "user", "content": NUDGE}


def test_no_nudge_when_disabled(store, client, fake):
    fake.script({"content": "Hello."})
    run = Agent(OllamaChatBackend(client), Toolbox(store), nudge=False).run("hi")
    assert run.answer == "Hello." and not run.steps


def test_repeated_identical_calls_are_stopped(store, client, fake):
    call = {"content": "", "tool_calls": [{"function": {"name": "today", "arguments": {}}}]}
    fake.script(call, call, call, call)
    run = Agent(OllamaChatBackend(client), Toolbox(store)).run("date?")
    assert run.stopped == "repeated_call" and len(run.steps) == 1
    assert "repeating" in run.answer


def test_step_limit(store, client, fake):
    fake.script(*[{"content": "", "tool_calls": [{"function": {"name": "calculator",
                                                             "arguments": {"expression": f"{i}+1"}}}]}
                  for i in range(10)])
    run = Agent(OllamaChatBackend(client), Toolbox(store), max_steps=3).run("count")
    assert run.stopped == "max_steps" and len(run.steps) == 3 and "step limit" in run.answer


def test_unknown_tool_from_model_is_survivable(store, client, fake):
    fake.script({"content": "", "tool_calls": [{"function": {"name": "delete_file", "arguments": {"path": "/"}}}]},
                {"content": "I can't delete files."})
    run = Agent(OllamaChatBackend(client), Toolbox(store)).run("delete the safety file")
    assert run.steps[0].result.startswith("ERROR: there is no tool") and run.answer == "I can't delete files."


def test_backend_failure_is_reported(store):
    class Broken:
        provider, model = "ollama", "x"

        def chat(self, *a):
            raise ConnectionError("down")

    run = Agent(Broken(), Toolbox(store)).run("hi")
    assert run.stopped == "error" and "down" in run.answer


def test_too_many_parallel_calls_are_capped(store):
    class Many:
        provider, model, n = "test", "m", 0

        def chat(self, system, messages, tools):
            self.n += 1
            if self.n == 1:
                return BackendReply("", [ToolCall("calculator", {"expression": f"{i}+0"}) for i in range(9)])
            return BackendReply("done", [])

    run = Agent(Many(), Toolbox(store), max_calls_per_step=4).run("x")
    assert len(run.steps) == 4 and run.answer == "done"


def test_openai_compatible_backend_against_ollama_v1(store, fake):
    import openai

    client = openai.OpenAI(base_url=fake.url + "/v1", api_key="ollama")
    backend = make_backend("openai-compatible", "qwen3:4b-instruct", client=client, base_url=fake.url + "/v1")
    run = Agent(backend, Toolbox(store)).run("What is 12 * 7?")
    assert run.steps[0].result.endswith("= 84") and run.stopped == "answered"
    assert backend.local
    sent = fake.calls("/v1/chat/completions")[1]["messages"]
    assert sent[-1]["role"] == "tool" and sent[-1]["tool_call_id"]
    assert isinstance(sent[-2]["tool_calls"][0]["function"]["arguments"], str)


def test_claude_backend_message_shapes(store):
    sent = []

    def create(**kw):
        sent.append(kw)
        if len(sent) == 1:
            return NS(content=[NS(type="text", text="Let me check."),
                               NS(type="tool_use", id="tu_1", name="calculator", input={"expression": "2+2"})],
                      usage=NS(input_tokens=10, output_tokens=5))
        return NS(content=[NS(type="text", text="It is 4.")], usage=NS(input_tokens=20, output_tokens=3))

    run = Agent(ClaudeBackend(NS(messages=NS(create=create))), Toolbox(store)).run("2+2?")
    assert run.answer == "It is 4." and run.input_tokens == 30
    assert sent[0]["tools"][0]["input_schema"]["type"] == "object"
    second = sent[1]["messages"]
    assert second[1]["content"][1] == {"type": "tool_use", "id": "tu_1", "name": "calculator",
                                       "input": {"expression": "2+2"}}
    assert second[2]["content"][0]["type"] == "tool_result" and second[2]["content"][0]["tool_use_id"] == "tu_1"


def test_make_backend_rejects_unknown():
    with pytest.raises(ValueError):
        make_backend("gemini")
