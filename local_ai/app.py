"""Wiring: build the client, store, answerer and agent from Settings. Shared by the CLI, the web
page and the scripts so they all behave the same way."""
from dataclasses import dataclass

from .agent import Agent, make_backend
from .config import Settings, load_key
from .llm import PROVIDERS, make_answerer
from .ollama import OllamaClient
from .privacy import check_local_host
from .rag import Assistant
from .store import Embedder, VectorStore
from .tools import Toolbox


@dataclass
class App:
    settings: Settings
    ollama: OllamaClient
    store: VectorStore
    assistant: Assistant
    agent: Agent

    @property
    def fully_local(self) -> bool:
        return self.settings.provider == "ollama" and self.settings.agent_provider == "ollama"


def build(settings: Settings, ollama: OllamaClient = None, cloud_client=None, agent_client=None) -> App:
    check_local_host(settings.ollama_host, settings.allow_remote_ollama)
    ollama = ollama or OllamaClient(settings.ollama_host)
    store = VectorStore(settings.index_path, Embedder(ollama, settings.embed_model), settings.screen_passages)
    if settings.provider == "ollama":
        answerer = make_answerer("ollama", settings.company, settings.model, settings.top_k,
                                 client=ollama, num_ctx=settings.num_ctx)
    else:
        key = PROVIDERS.get(settings.provider, (None,) * 5)[4]
        if key and cloud_client is None:
            load_key(key)
        answerer = make_answerer(settings.provider, settings.company, settings.model, settings.top_k,
                                 client=cloud_client)
    if settings.agent_provider == "ollama":
        backend = make_backend("ollama", settings.agent_model, ollama_client=ollama)
        backend.num_ctx = settings.num_ctx
    else:
        if agent_client is None and settings.agent_provider == "anthropic":
            load_key("ANTHROPIC_API_KEY")
        backend = make_backend(settings.agent_provider, settings.agent_model, client=agent_client)
    assistant = Assistant(store, answerer, settings.top_k, settings.min_score)
    agent = Agent(backend, Toolbox(store), settings.company)
    return App(settings, ollama, store, assistant, agent)
