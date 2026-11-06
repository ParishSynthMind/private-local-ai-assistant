"""Talk to the local model through Ollama's OpenAI-compatible endpoint, with a tool.

    python scripts/openai_compat_example.py [model]

The same three lines of setup work for any tool that speaks the OpenAI API (OpenCode, Continue,
LangChain, your own scripts): point base_url at Ollama and pass any api_key.
"""
import json
import sys

from openai import OpenAI

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:4b-instruct"
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")  # the key is required but ignored

tools = [{"type": "function", "function": {
    "name": "calculator", "description": "Do arithmetic exactly.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]

print("models:", [m.id for m in client.models.list().data])
reply = client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": "What is 380 * 0.08? Use the calculator."}], tools=tools)
choice = reply.choices[0]
print("finish_reason:", choice.finish_reason)
for call in choice.message.tool_calls or []:
    print("tool call:", call.function.name, json.loads(call.function.arguments))
print("tokens:", reply.usage.prompt_tokens, "in,", reply.usage.completion_tokens, "out")
