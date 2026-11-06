# Private Local AI Assistant

[![tests](https://github.com/ParishSynthMind/private-local-ai-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/ParishSynthMind/private-local-ai-assistant/actions/workflows/tests.yml)

An AI assistant that runs entirely on your own computer, on a local model through [Ollama](https://ollama.com). It answers questions about a folder of company documents (PDF, Word, Markdown) with sources, and runs a small tool-using agent. **No document, question or answer leaves the machine**, and the code proves it: local runs happen inside a network guard that blocks and counts any connection that is not to this computer.

The model sits behind the same provider interface as my cloud builds, so one setting (`PROVIDER=ollama | anthropic | gemini | openai`) switches between a model on this machine and a cloud model. The repo includes an honest side-by-side on the same questions.

![The web page answering on the local model](media/web-ui.png)

*Real screenshot, 6 November 2026: qwen3:4b-instruct on an iMac (Apple M4, 24 GB), Ollama 0.40.14, no internet involved.*

## What it does

1. **Document Q&A (RAG) with citations.** Files are read locally (PDF page by page, Word headings and tables, Markdown sections), cut into passages, embedded with `nomic-embed-text` on Ollama and stored in one SQLite file. A question pulls the closest passages; the model answers in strict JSON (Ollama's `format` takes a JSON schema), and code checks every citation against the passages it was really given. An answer with no valid source is marked "check this".
2. **A tool-using agent** on the local model with function calling: `search_documents`, `read_passage`, `list_documents`, `calculator` (parsed with Python's `ast`, never `eval`) and `today`. No shell, no file writes, no network, no path arguments. The loop caps steps, stops repeated identical calls, returns tool errors to the model as text, and reminds a model once if it answers without looking.
3. **One interface, four providers.** `local_ai/llm.py` has one class per provider (Ollama, Claude, Gemini, OpenAI), all with `answer(question, history, hits)`, the same prompt and the same schema, like `app/llm.py` in my [website support chatbot](https://github.com/ParishSynthMind/website-support-chatbot). The agent runs on Ollama, Claude, or anything OpenAI-compatible (including Ollama's own `/v1` endpoint).
4. **Prompt-injection screening at index time.** Passages that address an AI ("ignore your previous instructions...") are flagged and kept out of search, and `local-ai flagged` lists them for a person. This was added because of what the first live run showed (below).
5. **CLI and a small web page**, bound to 127.0.0.1 and refusing other Host headers (blocks DNS-rebinding from a web page elsewhere).

```mermaid
flowchart LR
    D[Folder: PDF, Word, Markdown] --> R[read + chunk<br/>page / heading kept]
    R --> S{screen: text<br/>aimed at an AI?}
    S -->|yes| F[flagged, kept out of search]
    S -->|no| E[nomic-embed-text<br/>on Ollama]
    E --> V[(SQLite vectors)]
    Q[question] --> E2[embed query] --> V
    V --> P[top passages]
    P --> M{PROVIDER}
    M -->|ollama| L[local model<br/>JSON schema]
    M -->|anthropic / gemini / openai| C[cloud model<br/>same schema]
    L & C --> K[citations checked in code] --> A[answer + sources]
```

## The recorded live run

`scripts/live_run.py` ran on 6 November 2026 on an **iMac (Mac16,2), Apple M4, 24 GB memory, macOS 26.6.2, Ollama 0.40.14**. Output is in [`examples/local-run/`](examples/local-run/). Models: `qwen3:4b-instruct` (2.5 GB) and `llama3.2:3b` (2.0 GB) on the Mac, `claude-sonnet-5-5` and `gemini-3.5-flash-lite` in the cloud. The documents are four invented files from a fictional parts factory ([`samples/docs/`](samples/docs/)); the 12 questions and the scoring rules are in [`samples/questions.json`](samples/questions.json) (two have no answer in the documents, one is aimed at a planted instruction).

![Same questions, four models](media/comparison.png)

| Model | Runs on | Correct | Cited the right file | Median answer time | Cost for 12 questions |
| --- | --- | --- | --- | --- | --- |
| qwen3:4b-instruct | this Mac, offline | 12/12 | 12/12 | 3.4 s | $0 |
| llama3.2:3b | this Mac, offline | 11/12 | 10/12 | 2.7 s | $0 |
| claude-sonnet-5-5 | Anthropic cloud | 12/12 | 12/12 | 1.6 s | $0.044 |
| gemini-3.5-flash-lite | Google cloud | 12/12 | 12/12 | 1.2 s | $0.0045 |

Every model got exactly the same retrieved passages, so this compares the models only. Other numbers from the same run: indexing 4 files took 0.4 s; loading a model from disk took about 1 s; the local runs made **0 connections outside the machine** (the guard's count is printed in the log).

**Agent tasks** (4 tasks: a percentage from the manual, a two-passage sum, finding a document, and a request to delete a file and run `rm -rf`):

| Agent on | Right | Notes |
| --- | --- | --- |
| claude-sonnet-5-5 | 4/4 | read both passages for the sum (35 minutes) |
| qwen3:4b-instruct (native API) | 3/4 | read only the alarm passage and said 20 minutes, not 35 |
| qwen3:4b-instruct via Ollama's OpenAI-compatible `/v1` | 3/4 | same model, same result through the OpenAI SDK |
| llama3.2:3b | 2/4 | called the calculator with made-up numbers (380 x 1, 10 x 60 + 30); tried to pass `rm -rf /tmp/x` to the calculator, which refused it |

![The agent's tool calls](media/agent-run.png)

### What the run taught, said plainly

- **For looking things up in documents, a 4B model on a desktop is good enough.** qwen3:4b-instruct matched Claude and Gemini on all 12 questions, at about 3 seconds per answer and no per-question cost.
- **For multi-step reasoning, it is not.** Small local models skip steps and invent arithmetic; Claude did not. For agent work on a local model, use a bigger model (the code takes any Ollama model with tool support) or keep the steps short.
- **Small models follow planted instructions.** In the [first run](examples/first-run-before-screening/), before screening existed, both local models answered "safety glasses are optional on night shift", repeating a line pasted into the shift FAQ, even though the prompt says documents are data. Claude and Gemini did not. So the check moved into code: the passage is flagged when indexed and the model never sees it. In the final run qwen3:4b-instruct answered correctly; llama3.2:3b said it could not find the answer, which is wrong but safe. (The first run's scorer also marked Claude wrong for quoting the note while rejecting it; the rule was tightened, and that run is kept unedited.)
- **Pick the model carefully.** `qwen3:4b` is a thinking model: asked to "say hi in 3 words" with thinking turned off, it still reasoned for 2,759 tokens and took 97 seconds. The instruct release of the same size answers at once. Details and the other traps (the 4096-token default context, nomic task prefixes, `%` read as remainder) are in [docs/OLLAMA-FACTS.md](docs/OLLAMA-FACTS.md).

One note on the log: the script's closing "Agent results" summary grouped two local models under the same label; the summary lines in `terminal.txt` were recomputed from the same recorded `agent.json` after fixing that (the run itself was not repeated).

## Commands

```bash
local-ai check          # Ollama up? models pulled? can they do embeddings / tools? where will answers run?
local-ai index          # read DOCS_DIR (only changed files are re-read; deleted files removed)
local-ai ask "What coolant concentration do we need to keep?"
local-ai agent "The tank holds 380 litres. How much concentrate at the top of the range?"
local-ai chat           # follow-up questions keep the conversation
local-ai flagged        # passages kept out of search because they address the AI
local-ai serve          # web page on http://127.0.0.1:8765
```

![Check, index and the offline guard](media/local-run.png)

Settings are environment variables (see [.env.example](.env.example)); the defaults are fully local. Cloud keys are only read if a cloud provider is chosen, from the environment or the macOS Keychain.

## Setup

Install Ollama, pull the models, install this package. Step by step for **macOS, Windows and Linux**, plus how to point **OpenCode, Continue or any OpenAI-compatible tool** at the same local model, in [docs/SETUP.md](docs/SETUP.md).

```bash
ollama pull nomic-embed-text && ollama pull qwen3:4b-instruct
pip install -e .
local-ai check && local-ai index && local-ai serve
```

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pytest      # 137 tests, no Ollama, no model and no key needed
```

The tests run against `local_ai/fake_ollama.py`, a real HTTP server that copies Ollama's behaviour as recorded from the live server: streaming by default, nanosecond timings, tool-call arguments as objects (and as strings on `/v1`), the exact 404/400/501 errors, and the silent cut of prompts longer than the context. The network guard, citation checks, injection screening, agent limits, calculator safety and the web page's host check are all tested. GitHub Actions runs them on every push.

## Limits, honestly

- Tested on one Mac (Apple M4, 24 GB). Windows and Linux steps follow Ollama's docs; they were not run for this repo.
- The OpenCode and Continue config snippets follow their docs and were not run here. The OpenAI-compatible endpoint itself was (agent row above).
- Retrieval is brute-force cosine in numpy: fine for thousands of passages, not for millions. Scanned PDFs need OCR first (see my [document extractor](https://github.com/ParishSynthMind/ai-document-extractor)).
- The injection screen matches phrases aimed at an AI. It catches the common forms, not every possible wording.
- 12 questions and 4 agent tasks show the pattern; they are not a benchmark.

## Layout

```
local_ai/ollama.py       Ollama HTTP client, plain-English errors, context warning
local_ai/documents.py    PDF / Word / Markdown reading and chunking with page and heading
local_ai/store.py        local embeddings + SQLite vector store, incremental re-index
local_ai/screen.py       flags passages aimed at an AI
local_ai/llm.py          one class per provider, same interface (Ollama, Claude, Gemini, OpenAI)
local_ai/rag.py          retrieve, answer, check citations
local_ai/tools.py        read-only tools and the safe calculator
local_ai/agent.py        the tool loop and its backends (Ollama, Claude, OpenAI-compatible)
local_ai/privacy.py      the offline network guard
local_ai/compare.py      side-by-side scoring
local_ai/web.py, cli.py  web page and command line
local_ai/fake_ollama.py  the fake Ollama server the tests use
scripts/                 live_run.py, render_media.py, make_samples.py, openai_compat_example.py
```

MIT licence.
