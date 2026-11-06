"""Question -> local retrieval -> model answer -> citations checked in code.

The model returns citation ids; code, not the model, decides whether they are real:
ids that point at no passage are dropped, and an "answered" reply with no valid citation
is flagged as ungrounded so the UI can say "check this" instead of presenting it as fact.
"""
import time
from dataclasses import dataclass, field
from typing import List

from .llm import AnswerResult
from .store import Hit, VectorStore


@dataclass
class Citation:
    id: int
    source: str
    page: int
    heading: str
    label: str
    quote: str


@dataclass
class Answer:
    question: str
    status: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    grounded: bool = True
    dropped_citations: List[int] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    answer_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    local: bool = False
    hits: List[Hit] = field(default_factory=list)

    @property
    def sources(self) -> List[str]:
        return sorted({c.source for c in self.citations})

    def to_dict(self) -> dict:
        return {
            "question": self.question, "status": self.status, "answer": self.text, "grounded": self.grounded,
            "citations": [c.__dict__ for c in self.citations], "dropped_citations": self.dropped_citations,
            "retrieval_seconds": round(self.retrieval_seconds, 3), "answer_seconds": round(self.answer_seconds, 3),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "provider": self.provider, "model": self.model, "local": self.local,
        }


NOTHING_FOUND = "I could not find that in the documents."


class Assistant:
    def __init__(self, store: VectorStore, answerer, top_k: int = 5, min_score: float = 0.35):
        self.store, self.answerer, self.top_k, self.min_score = store, answerer, top_k, min_score

    def retrieve(self, question: str) -> List[Hit]:
        return self.store.search(question, k=self.top_k, min_score=self.min_score)

    def ask(self, question: str, history: list = None, hits: List[Hit] = None) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")
        t0 = time.monotonic()
        if hits is None:
            hits = self.retrieve(question)
        t1 = time.monotonic()
        meta = dict(provider=self.answerer.provider, model=self.answerer.model, local=self.answerer.local)
        if not hits:  # nothing close enough: do not even ask the model, it would only guess
            return Answer(question, "not_in_documents", NOTHING_FOUND, retrieval_seconds=t1 - t0, **meta)
        result: AnswerResult = self.answerer.answer(question, history or [], hits)
        return self._check(question, result, hits, t1 - t0, **meta)

    @staticmethod
    def _check(question: str, result: AnswerResult, hits: List[Hit], retrieval_seconds: float, **meta) -> Answer:
        valid, dropped, seen = [], [], set()
        for cid in result.citations:
            if 1 <= cid <= len(hits) and cid not in seen:
                seen.add(cid)
                h = hits[cid - 1]
                valid.append(Citation(cid, h.chunk.source, h.chunk.page, h.chunk.heading, h.chunk.label,
                                      h.chunk.text[:240]))
            elif cid not in seen:
                dropped.append(cid)
        status = result.status
        if status == "not_in_documents":
            valid = []  # a refusal carries no sources
        grounded = status != "answered" or bool(valid)
        return Answer(question, status, result.answer or NOTHING_FOUND, valid, grounded, dropped,
                      retrieval_seconds, result.seconds, result.input_tokens, result.output_tokens,
                      hits=hits, **meta)
