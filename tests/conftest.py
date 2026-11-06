import shutil
from pathlib import Path

import pytest

from local_ai.fake_ollama import FakeOllama
from local_ai.ollama import OllamaClient
from local_ai.store import Embedder, VectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples" / "docs"


@pytest.fixture
def fake():
    with FakeOllama() as f:
        yield f


@pytest.fixture
def client(fake):
    return OllamaClient(fake.url, timeout=10, sleep=lambda s: None)


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    shutil.copytree(SAMPLES, d)
    return d


@pytest.fixture
def store(client, tmp_path, docs):
    s = VectorStore(tmp_path / "index.sqlite", Embedder(client))
    s.index_folder(docs)
    return s
