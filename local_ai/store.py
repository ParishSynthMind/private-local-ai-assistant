"""Local embeddings and a small vector store in one SQLite file.

Embeddings always come from Ollama on this machine (default nomic-embed-text), whichever
model writes the answers, so the documents are never sent anywhere to be indexed.

The store keeps one row per chunk with its vector as float32 bytes and searches by cosine
similarity in numpy. Brute force is honest and fast at this size: a few thousand chunks
search in about a millisecond. For hundreds of thousands, swap in sqlite-vec or Chroma
behind the same search() method.

Re-indexing is incremental: a file is only re-read when its sha256 changes, deleted files
are removed, and an index built with a different embedding model is refused (vectors from
two models are not comparable, and mixing them quietly ruins retrieval).

Passages that look like instructions aimed at an AI are flagged while indexing (screen.py) and
left out of search results, so the model never sees them; `flagged()` lists them for review.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .documents import Chunk, chunk_document, file_hash, find_documents
from .screen import screen

# nomic-embed-text was trained with task prefixes; leaving them off measurably hurts retrieval.
PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}


class IndexMismatch(RuntimeError):
    pass


class Embedder:
    def __init__(self, client, model: str = "nomic-embed-text"):
        self.client = client
        self.model = model
        base = model.split(":")[0]
        self.doc_prefix, self.query_prefix = PREFIXES.get(base, ("", ""))

    def documents(self, texts: List[str]) -> np.ndarray:
        return _matrix(self.client.embed(self.model, [self.doc_prefix + t for t in texts]))

    def query(self, text: str) -> np.ndarray:
        return _matrix(self.client.embed(self.model, [self.query_prefix + text]))[0]


def _matrix(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms  # already unit length from /api/embed, but other servers may not be


@dataclass
class Hit:
    chunk: Chunk
    score: float
    chunk_id: int


@dataclass
class IndexReport:
    added: List[str]
    updated: List[str]
    unchanged: List[str]
    removed: List[str]
    failed: Dict[str, str]
    chunks: int
    seconds: float
    flagged: List[str] = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha256 TEXT, chunks INTEGER, indexed_at REAL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, path TEXT, heading TEXT, page INTEGER, position INTEGER, text TEXT, vector BLOB,
    flag TEXT);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""


class VectorStore:
    def __init__(self, db_path: Path, embedder: Embedder, screen_passages: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        if "flag" not in [r[1] for r in self.db.execute("PRAGMA table_info(chunks)")]:
            self.db.execute("ALTER TABLE chunks ADD COLUMN flag TEXT")  # index made before screening existed
        self.embedder = embedder
        self.screen_passages = screen_passages
        self._cache = None  # (ids, matrix) loaded on first search

    # --- meta -----------------------------------------------------------------------------
    def _meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def check_model(self):
        built = self._meta("embed_model")
        if built and built != self.embedder.model:
            raise IndexMismatch(f"This index was built with {built!r} but the embedder is {self.embedder.model!r}. "
                                f"Rebuild it with `local-ai index --rebuild`.")

    # --- indexing -------------------------------------------------------------------------
    def index_folder(self, root: Path, rebuild: bool = False, max_chars: int = 900, overlap: int = 150,
                     progress: Callable[[str], None] = lambda s: None) -> IndexReport:
        started = time.monotonic()
        root = Path(root)
        if rebuild:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM files")
            self.db.execute("DELETE FROM meta")
        self.check_model()
        known = {p: h for p, h in self.db.execute("SELECT path, sha256 FROM files")}
        added, updated, unchanged, failed, flagged = [], [], [], {}, []
        seen = set()
        for path in find_documents(root):
            rel = path.relative_to(root).as_posix()
            seen.add(rel)
            digest = file_hash(path)
            if known.get(rel) == digest:
                unchanged.append(rel)
                continue
            try:
                chunks = chunk_document(path, root, max_chars, overlap)
                vectors = self.embedder.documents([f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]) if chunks else np.zeros((0, 0))
            except Exception as e:  # a broken file must not stop the rest of the folder
                failed[rel] = f"{type(e).__name__}: {e}"
                progress(f"  could not read {rel}: {failed[rel]}")
                continue
            with self.db:
                self.db.execute("DELETE FROM chunks WHERE path=?", (rel,))
                for chunk, vec in zip(chunks, vectors):
                    flag = screen(chunk.text) if self.screen_passages else None
                    if flag:
                        flagged.append(f"{chunk.label}: {flag!r}")
                        progress(f"  FLAGGED {chunk.label}: looks like instructions to an AI ({flag!r}); left out of search")
                    self.db.execute(
                        "INSERT INTO chunks (path, heading, page, position, text, vector, flag) VALUES (?,?,?,?,?,?,?)",
                        (rel, chunk.heading, chunk.page, chunk.position, chunk.text, vec.astype(np.float32).tobytes(),
                         flag))
                self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?)", (rel, digest, len(chunks), time.time()))
                self._set_meta("embed_model", self.embedder.model)
                if len(vectors):
                    self._set_meta("dim", str(vectors.shape[1]))
            (updated if rel in known else added).append(rel)
            progress(f"  {'updated' if rel in known else 'indexed'} {rel}: {len(chunks)} chunks")
        removed = sorted(set(known) - seen)
        with self.db:
            for rel in removed:
                self.db.execute("DELETE FROM chunks WHERE path=?", (rel,))
                self.db.execute("DELETE FROM files WHERE path=?", (rel,))
                progress(f"  removed {rel} (no longer in the folder)")
        self._cache = None
        total = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return IndexReport(added, updated, unchanged, removed, failed, total, time.monotonic() - started, flagged)

    # --- reading --------------------------------------------------------------------------
    def files(self) -> List[dict]:
        return [{"path": p, "chunks": n} for p, n in self.db.execute("SELECT path, chunks FROM files ORDER BY path")]

    def flagged(self) -> List[dict]:
        return [{"id": i, "source": p, "heading": h, "page": pg, "reason": f, "text": t} for i, p, h, pg, f, t in
                self.db.execute("SELECT id, path, heading, page, flag, text FROM chunks WHERE flag IS NOT NULL")]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get(self, chunk_id: int) -> Optional[Chunk]:
        row = self.db.execute("SELECT path, heading, page, text, position FROM chunks WHERE id=?",
                              (int(chunk_id),)).fetchone()
        return Chunk(row[0], row[1], row[2], row[3], row[4]) if row else None

    def search(self, query: str, k: int = 5, min_score: float = 0.0, source: Optional[str] = None) -> List[Hit]:
        self.check_model()
        if self._cache is None:
            rows = self.db.execute("SELECT id, vector FROM chunks WHERE flag IS NULL ORDER BY id").fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1))
            self._cache = (ids, matrix)
        ids, matrix = self._cache
        if not len(ids):
            return []
        scores = matrix @ self.embedder.query(query)
        hits = []
        for i in np.argsort(-scores):
            if scores[i] < min_score:
                break
            chunk = self.get(int(ids[i]))
            if source and chunk.source != source:
                continue
            hits.append(Hit(chunk, float(scores[i]), int(ids[i])))
            if len(hits) >= k:
                break
        return hits
