import sqlite3

import numpy as np
import pytest

from local_ai.documents import chunk_document, find_documents, read_sections, split_text
from local_ai.store import Embedder, IndexMismatch, VectorStore


def test_finds_supported_files_and_skips_hidden_and_lock_files(docs):
    (docs / ".hidden.md").write_text("# x\nsecret")
    (docs / "~$lock.docx").write_bytes(b"x")
    (docs / "image.png").write_bytes(b"x")
    names = [p.name for p in find_documents(docs)]
    assert names == ["HX-500-Maintenance-Manual.pdf", "QP-07-Quality-Procedure.docx", "Safety-Rules.md",
                     "Shift-Handover-FAQ.md"]


def test_pdf_keeps_page_numbers(docs):
    sections = read_sections(docs / "HX-500-Maintenance-Manual.pdf")
    assert [s.page for s in sections] == [1, 2, 3]
    assert "6% and 8%" in sections[1].text and sections[1].heading == "Coolant"


def test_docx_reads_headings_and_tables(docs):
    sections = read_sections(docs / "QP-07-Quality-Procedure.docx")
    heads = [s.heading for s in sections]
    assert "First-article inspection" in heads and "Nonconformance" in heads
    table = [s for s in sections if "Bore gauge" in s.text][0]
    assert "6 months" in table.text


def test_markdown_split_by_heading(docs):
    sections = read_sections(docs / "Safety-Rules.md")
    assert [s.heading for s in sections] == ["Personal protective equipment", "Lockout-tagout",
                                             "Forklifts and walkways", "Emergencies"]


def test_chunk_labels(docs):
    pdf = chunk_document(docs / "HX-500-Maintenance-Manual.pdf", docs)
    md = chunk_document(docs / "Safety-Rules.md", docs)
    assert pdf[1].label == "HX-500-Maintenance-Manual.pdf, page 2"
    assert md[0].label == "Safety-Rules.md, Personal protective equipment"


def test_split_text_respects_limit_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = split_text(text, max_chars=300, overlap=60)
    assert len(pieces) > 5 and all(len(p) <= 300 for p in pieces)
    assert pieces[1].split()[0] in pieces[0]  # the next piece starts inside the previous one


def test_split_text_handles_a_line_with_no_punctuation():
    pieces = split_text("x" * 2500, max_chars=900, overlap=100)
    assert all(len(p) <= 900 for p in pieces) and len(pieces) >= 3


def test_split_text_short_and_empty():
    assert split_text("short") == ["short"]
    assert split_text("   ") == []


def test_index_and_search_finds_the_right_file(store):
    hits = store.search("What coolant concentration do we need to keep?", k=3)
    assert hits[0].chunk.source == "HX-500-Maintenance-Manual.pdf" and hits[0].chunk.page == 2
    assert hits[0].score >= hits[-1].score


def test_search_min_score_and_source_filter(store):
    assert store.search("zebra quantum banana", k=5, min_score=0.5) == []
    hits = store.search("safety glasses", k=5, source="Safety-Rules.md")
    assert hits and all(h.chunk.source == "Safety-Rules.md" for h in hits)


def test_reindex_skips_unchanged_files(store, docs, fake):
    before = len(fake.calls("/api/embed"))
    rep = store.index_folder(docs)
    assert len(rep.unchanged) == 4 and not rep.added and len(fake.calls("/api/embed")) == before


def test_reindex_updates_changed_and_removes_deleted(store, docs):
    (docs / "Safety-Rules.md").write_text("# Safety\n## Forklifts\nForklifts must not exceed 8 km/h.")
    (docs / "Shift-Handover-FAQ.md").unlink()
    rep = store.index_folder(docs)
    assert rep.updated == ["Safety-Rules.md"] and rep.removed == ["Shift-Handover-FAQ.md"]
    assert "8 km/h" in store.search("forklift speed", k=1)[0].chunk.text
    assert all(f["path"] != "Shift-Handover-FAQ.md" for f in store.files())


def test_broken_file_does_not_stop_the_folder(client, tmp_path, docs):
    (docs / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    s = VectorStore(tmp_path / "i.sqlite", Embedder(client))
    rep = s.index_folder(docs)
    assert "broken.pdf" in rep.failed and len(rep.added) == 4


def test_index_built_with_other_model_is_refused(client, tmp_path, docs):
    s = VectorStore(tmp_path / "i.sqlite", Embedder(client, "nomic-embed-text"))
    s.index_folder(docs)
    other = VectorStore(tmp_path / "i.sqlite", Embedder(client, "mxbai-embed-large"))
    with pytest.raises(IndexMismatch, match="--rebuild"):
        other.search("coolant")


def test_rebuild_clears_everything(store, docs):
    rep = store.index_folder(docs, rebuild=True)
    assert len(rep.added) == 4 and rep.chunks == store.count()


def test_nomic_prefixes_are_sent(client, fake):
    e = Embedder(client, "nomic-embed-text")
    e.query("coolant")
    e.documents(["a passage"])
    sent = [b["input"][0] for b in fake.calls("/api/embed")]
    assert sent == ["search_query: coolant", "search_document: a passage"]


def test_vectors_stored_as_float32(store):
    db = sqlite3.connect(str(store.db_path))
    blob = db.execute("SELECT vector FROM chunks LIMIT 1").fetchone()[0]
    assert np.frombuffer(blob, dtype=np.float32).shape == (768,)


def test_empty_index_search(client, tmp_path):
    s = VectorStore(tmp_path / "empty.sqlite", Embedder(client))
    assert s.search("anything") == [] and s.count() == 0
