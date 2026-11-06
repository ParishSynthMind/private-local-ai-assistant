# Ollama facts this project relies on

Checked by calling a real Ollama 0.40.14 on an iMac (Apple M4, 24 GB) on 6 November 2026, unless a line says it comes from the docs. The fake server in `local_ai/fake_ollama.py` copies every behaviour below, and the tests hold the client to it.

## Requests and responses

| What | Real behaviour |
| --- | --- |
| `/api/chat` streaming | Streams NDJSON **by default**. Send `"stream": false` for one JSON reply. |
| Durations | Nanoseconds: `total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`. Token counts are `prompt_eval_count` and `eval_count`. |
| Tool calls | `message.tool_calls[].function.arguments` is a JSON **object** on `/api/chat`, and a JSON **string** on `/v1/chat/completions` (which also sets `finish_reason: "tool_calls"`). Each call has an `id`. |
| Tool results | Sent back as `{"role": "tool", "content": "...", "tool_name": "calculator"}` on `/api/chat`. |
| Structured output | `"format": <JSON schema>` constrains decoding. Even a 3B model then returns valid JSON. |
| Embeddings | `/api/embed` takes a string or a list and returns unit-length vectors (nomic-embed-text: 768 numbers). |
| Capabilities | `/api/tags` and `/api/show` list them: `completion`, `tools`, `thinking`, `embedding`. |

## Errors (exact text)

| Situation | Status | Body |
| --- | --- | --- |
| Model not pulled (chat, embed) | 404 | `{"error":"model 'nope:1b' not found"}` |
| Model not pulled (`/api/show`, bare name) | 404 | `{"error":"model 'nope:latest' not found"}` |
| Embeddings from a chat model | 501 | `This server does not support embeddings. Start it with --embeddings` |
| Chat with an embedding model | 400 | `"nomic-embed-text" does not support chat` |
| `"think": true` on a model without thinking | 400 | `"llama3.2:3b" does not support thinking` |
| Tools on a model without the tools capability | 400 | `"<model>" does not support tools` (same pattern; not reproduced live, as every pulled chat model here supports tools) |
| Nothing listening | connection refused | the client turns this into "start it with `ollama serve`" |

## Things that quietly go wrong

1. **The context window is 4096 tokens unless you ask for more.** `ollama ps` showed `context_length: 4096` for a model loaded without `num_ctx`, on this 24 GB Mac. A RAG prompt longer than that is cut from the front with no error, which can remove the system prompt and the first passages. This project always sends `options.num_ctx` (8192 by default) and warns when a prompt uses 90% of it. Ollama's docs say the default depends on memory (4k under 24 GiB of VRAM, 32k from 24 to 48 GiB); on Apple silicon the GPU share of unified memory is below the total, so 24 GB of RAM still got 4k.
2. **`qwen3:4b` is a thinking model, and `"think": false` did not stop it.** Its chat template always opens a `<think>` block. Asked to "say hi in 3 words" with think false, it wrote 2,759 tokens of reasoning into the content and took **96.9 seconds** before answering "Hi what's up". `qwen3:4b-instruct` (the non-thinking release, same size) answers the same prompt at once, so it is the default here. The client also strips anything before a closing `</think>` in case a thinking model is chosen.
3. **Small models write `%` as Python's remainder operator.** Asked for "17.5% of 2340" with a calculator tool, qwen3:4b-instruct called it with `"17.5 % 2340"`, which Python reads as 17.5 remainder 2340 = 17.5. The calculator here refuses a bare `a % b` and accepts `17.5% of 2340`.
4. **Small models sometimes answer without calling any tool.** In the first live agent run, qwen3:4b-instruct replied "I cannot determine..." without searching. The agent loop now reminds it once to search; on the next step it searched, and answered 30.4 litres correctly.
5. **nomic-embed-text expects task prefixes** (`search_document: ` and `search_query: `). They are added automatically.
6. **An index built with one embedding model is useless with another.** The store records the model and refuses to search with a different one.

## From the Ollama docs (fetched 6 November 2026)

- Ollama binds `127.0.0.1:11434` by default; `OLLAMA_HOST` changes it. Do not set it to `0.0.0.0` unless you mean to share the server with your network.
- Models live in `~/.ollama/models` (macOS), `/usr/share/ollama/.ollama/models` (Linux service) and `C:\Users\<you>\.ollama\models` (Windows); `OLLAMA_MODELS` moves them.
- `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` sets a server-wide default context; `options.num_ctx` sets it per request.
- `OLLAMA_NO_CLOUD=1` (or `{"disable_ollama_cloud": true}` in `~/.ollama/server.json`) turns off Ollama's cloud features for local-only use.
- "Ollama runs locally. We don't see your prompts or data when you run locally." Internet is needed to pull models. The desktop apps for macOS and Windows download updates automatically.
- The OpenAI-compatible API is at `http://localhost:11434/v1/` (any API key, it is ignored): chat completions with tools and streaming, completions, models, embeddings, and `/v1/responses` from 0.13.3. It has no way to set the context size; build a model with `PARAMETER num_ctx` in a Modelfile for tools that only speak that API.
