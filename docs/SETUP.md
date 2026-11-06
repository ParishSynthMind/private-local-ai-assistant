# Setup on macOS, Windows and Linux

The assistant needs two things: **Ollama** (runs the models) and **Python 3.10+** (runs this project). About 5 GB of disk for the default models, and 8 GB of RAM as a practical minimum (16 GB is comfortable).

## 1. Install Ollama

| System | How |
| --- | --- |
| macOS 14+ | Download the app from ollama.com (drag to Applications), or `brew install ollama`. |
| Windows 10 22H2+ | Run `OllamaSetup.exe` from ollama.com (no admin rights needed). |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` (installs a systemd service). |

The desktop apps start Ollama for you. With Homebrew or a manual install, start it yourself in a terminal you leave open:

```bash
ollama serve
```

It listens on `127.0.0.1:11434` only, so nothing on your network can reach it. Check with `curl http://127.0.0.1:11434/api/version`.

**Private by default, private on purpose.** To also switch off Ollama's optional cloud features, set `OLLAMA_NO_CLOUD=1` before starting it (macOS app: `launchctl setenv OLLAMA_NO_CLOUD 1`, then restart the app; Linux service: `systemctl edit ollama.service` and add `Environment="OLLAMA_NO_CLOUD=1"`; Windows: add it under "Edit environment variables for your account", then restart Ollama).

## 2. Pull the models

```bash
ollama pull nomic-embed-text     # 0.27 GB, embeddings for search
ollama pull qwen3:4b-instruct    # 2.5 GB, answers and tool calling (default)
ollama pull llama3.2:3b          # 2.0 GB, optional second model for the comparison
```

Bigger models answer better and need more memory: roughly, a 7-8B model at the default 4-bit size wants 8 GB free, a 14B model 16 GB. Change the model with `MODEL=` and `AGENT_MODEL=`; `local-ai check` tells you if it is missing or cannot call tools.

Where models are stored: `~/.ollama/models` (macOS), `/usr/share/ollama/.ollama/models` (Linux service), `C:\Users\<you>\.ollama\models` (Windows). Remove one with `ollama rm <name>`.

## 3. Install and run the assistant

macOS / Linux:

```bash
git clone https://github.com/ParishSynthMind/private-local-ai-assistant
cd private-local-ai-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
local-ai check
local-ai index
local-ai ask "What coolant concentration do we need to keep?"
local-ai serve            # http://127.0.0.1:8765
```

Windows (PowerShell):

```powershell
git clone https://github.com/ParishSynthMind/private-local-ai-assistant
cd private-local-ai-assistant
py -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e .
local-ai check
```

To use your own documents, point `DOCS_DIR` at a folder of PDF, Word, Markdown or text files and run `local-ai index` again. Only changed files are re-read.

## 4. Use the same local model from coding tools (OpenAI-compatible endpoint)

Ollama also speaks the OpenAI Chat Completions API at `http://127.0.0.1:11434/v1`. Any API key works (it is ignored). This is how editors and agents such as OpenCode and Continue use a local model.

**Raise the context first.** That API has no way to set the context size, and coding agents need far more than the 4096-token default for their tool definitions. Make a copy of the model with a bigger window:

```bash
printf 'FROM qwen3:4b-instruct\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen3-4b-32k -f Modelfile
```

(OpenCode's docs suggest 16k to 32k if tool calls are not working; Ollama's docs suggest 64k for agents and coding tools if you have the memory.)

**Python** (`scripts/openai_compat_example.py`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
reply = client.chat.completions.create(model="qwen3-4b-32k", messages=[{"role": "user", "content": "Hello"}])
```

**OpenCode** (`opencode.json` in the project, or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "qwen3-4b-32k": { "name": "Qwen3 4B, 32k context" } }
    }
  }
}
```

The model id must match what `ollama list` (or `GET /v1/models`) shows.

**Continue** (`config.yaml`, under `~/.continue/`):

```yaml
name: Local
version: 0.0.1
schema: v1
models:
  - name: Qwen3 4B local
    provider: ollama
    model: qwen3-4b-32k
    apiBase: http://localhost:11434
    roles: [chat, edit]
    capabilities: [tool_use]
```

For a fully offline editor, also turn off Continue's "Allow Anonymous Telemetry" setting.

These two config snippets follow the OpenCode and Continue docs as read on 6 November 2026; they were not run in this project (the endpoint itself was, see the README).

## Troubleshooting

| Message | Fix |
| --- | --- |
| `Ollama is not running at http://127.0.0.1:11434` | Start it: `ollama serve`, or open the Ollama app. |
| `Model 'x' is not on this machine` | `ollama pull x` |
| `does not support tools` | Pick a model whose `ollama show <model>` lists `tools`. |
| `This index was built with ...` | You changed `EMBED_MODEL`: run `local-ai index --rebuild`. |
| `OLLAMA_HOST points at ... not this machine` | Intended only for your own server: set `ALLOW_REMOTE_OLLAMA=1`. |
| Answers ignore the start of long documents | Raise `NUM_CTX` (the warning `prompt used N of M context tokens` tells you when). |
| Very slow first answer | The model is loading into memory; later answers are faster. Ollama unloads it after 5 idle minutes. |
