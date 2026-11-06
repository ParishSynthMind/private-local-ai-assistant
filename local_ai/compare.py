"""Side-by-side: the same questions, the same retrieved passages, different models.

Retrieval runs once per question (local embeddings), and every model gets exactly those passages,
so the comparison measures the models and nothing else. Each answer is scored by simple, visible
rules from samples/questions.json, not by another model:

- correct:  for an answerable question, status "answered" and every expected fact present
            (each fact is a list of accepted spellings, e.g. ["30%", "30 percent"]);
            for an out-of-scope question, status "not_in_documents".
- cited:    at least one citation points at an expected source file.
- forbidden: the answer contains a phrase it must not (e.g. repeating a planted instruction).
Timing is wall-clock on the machine that ran it; cost uses list prices from llm.PROVIDERS.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .llm import cost_usd
from .rag import Assistant


@dataclass
class Question:
    id: str
    question: str
    expect_facts: List[List[str]] = field(default_factory=list)
    expect_sources: List[str] = field(default_factory=list)
    answerable: bool = True
    forbidden: List[str] = field(default_factory=list)


def load_questions(path: Path) -> List[Question]:
    return [Question(**q) for q in json.loads(Path(path).read_text())]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("±", "+/-")).strip()


def score(q: Question, ans) -> dict:
    text = _norm(ans.text)
    if q.answerable:
        facts = all(any(_norm(alt) in text for alt in fact) for fact in q.expect_facts)
        correct = ans.status == "answered" and facts
    else:
        correct = ans.status == "not_in_documents"
    cited = (not q.answerable) or any(s in q.expect_sources for s in ans.sources)
    bad = [f for f in q.forbidden if _norm(f) in text]
    return {"correct": bool(correct) and not bad, "cited": bool(cited), "forbidden_hit": bad}


def run_comparison(store, answerers: list, questions: List[Question], top_k: int = 5, min_score: float = 0.35,
                   progress=lambda s: None) -> dict:
    rows = []
    for q in questions:
        hits = store.search(q.question, k=top_k, min_score=min_score)
        for answerer in answerers:
            assistant = Assistant(store, answerer, top_k, min_score)
            try:
                ans = assistant.ask(q.question, hits=hits)
                s = score(q, ans)
                row = dict(ans.to_dict(), qid=q.id, **s,
                           cost_usd=cost_usd(answerer.provider, ans.input_tokens, ans.output_tokens))
            except Exception as e:  # one bad reply must not lose the whole table
                row = {"qid": q.id, "question": q.question, "provider": answerer.provider, "model": answerer.model,
                       "status": "error", "answer": f"{type(e).__name__}: {e}", "correct": False, "cited": False,
                       "forbidden_hit": [], "answer_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                       "cost_usd": 0.0, "local": answerer.local}
            rows.append(row)
            progress(f"{q.id} {answerer.model}: {'OK ' if row['correct'] else 'X  '}"
                     f"{row['answer_seconds']:.1f}s {row['answer'][:70]}")
    return {"rows": rows, "summary": summarise(rows, answerers)}


def summarise(rows: List[dict], answerers: list) -> List[dict]:
    out = []
    for a in answerers:
        mine = [r for r in rows if r["model"] == a.model and r["provider"] == a.provider]
        if not mine:
            continue
        times = sorted(r["answer_seconds"] for r in mine)
        out.append({
            "provider": a.provider, "model": a.model, "local": a.local, "questions": len(mine),
            "correct": sum(r["correct"] for r in mine), "cited": sum(r["cited"] for r in mine),
            "forbidden_hits": sum(bool(r["forbidden_hit"]) for r in mine),
            "errors": sum(r["status"] == "error" for r in mine),
            "median_seconds": times[len(times) // 2], "max_seconds": times[-1],
            "tokens_in": sum(r["input_tokens"] for r in mine), "tokens_out": sum(r["output_tokens"] for r in mine),
            "cost_usd": round(sum(r["cost_usd"] for r in mine), 5),
        })
    return out


def markdown_table(summary: List[dict]) -> str:
    lines = ["| Model | Where it runs | Correct | Cited the right file | Median time | Slowest | Tokens in / out | Cost |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summary:
        lines.append(f"| {s['model']} | {'this Mac, offline' if s['local'] else s['provider'] + ' cloud'} | "
                     f"{s['correct']}/{s['questions']} | {s['cited']}/{s['questions']} | {s['median_seconds']:.1f}s | "
                     f"{s['max_seconds']:.1f}s | {s['tokens_in']:,} / {s['tokens_out']:,} | "
                     f"{'$0 (local)' if s['local'] else '$' + format(s['cost_usd'], '.4f')} |")
    return "\n".join(lines)
