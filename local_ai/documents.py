"""Read a folder of documents (PDF, Word, Markdown, plain text) and cut them into chunks.

Each chunk remembers where it came from (file, page for PDFs, heading for Word and
Markdown) so an answer can cite "CNC-Maintenance-Manual.pdf, page 2" rather than a
vague "the manual". Everything here runs locally with pure-Python libraries.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class Section:
    """A piece of a document before chunking: a PDF page or a run of text under one heading."""
    heading: str
    text: str
    page: int = 0  # 1-based for PDFs, 0 when the format has no pages


@dataclass
class Chunk:
    source: str      # path relative to the documents folder, e.g. "CNC-Maintenance-Manual.pdf"
    heading: str
    page: int
    text: str
    position: int    # order within the file

    @property
    def label(self) -> str:
        where = f"page {self.page}" if self.page else (self.heading or "start")
        return f"{self.source}, {where}"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def find_documents(folder: Path) -> List[Path]:
    """Supported files under folder, sorted, skipping hidden files and Office lock files (~$x.docx)."""
    out = []
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts) or p.name.startswith("~$"):
            continue
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            out.append(p)
    return out


def read_sections(path: Path) -> List[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in (".md", ".markdown"):
        return _read_markdown(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".txt":
        return [Section("", path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"unsupported file type: {path.name}")


def _read_pdf(path: Path) -> List[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            sections.append(Section(first[:80], text, page=i))
    return sections


def _read_docx(path: Path) -> List[Section]:
    import docx

    document = docx.Document(str(path))
    sections: List[Section] = []
    heading, lines = "", []

    def flush():
        if any(x.strip() for x in lines):
            sections.append(Section(heading, "\n".join(lines)))

    for para in document.paragraphs:
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading") or style == "title":
            flush()
            heading, lines = para.text.strip(), []
        elif para.text.strip():
            lines.append(para.text.strip())
    flush()
    # Tables hold a lot of the facts in real procedures, so read them too, row by row.
    for table in document.tables:
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        if rows:
            sections.append(Section(heading or "Table", "\n".join(rows)))
    return sections


def _read_markdown(text: str) -> List[Section]:
    sections: List[Section] = []
    heading, lines = "", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if any(x.strip() for x in lines):
                sections.append(Section(heading, "\n".join(lines).strip()))
            heading, lines = m.group(2).strip(), []
        else:
            lines.append(line)
    if any(x.strip() for x in lines):
        sections.append(Section(heading, "\n".join(lines).strip()))
    return sections


def split_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """Split on paragraph, then sentence boundaries, into pieces of at most max_chars.
    Consecutive pieces share about `overlap` characters so a fact on a boundary is not lost."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    units: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            for sent in re.split(r"(?<=[.!?])\s+|\n", para):
                sent = sent.strip()
                while len(sent) > max_chars:  # a monster line with no punctuation
                    units.append(sent[:max_chars])
                    sent = sent[max_chars - overlap:]
                if sent:
                    units.append(sent)
    pieces, current = [], ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_chars:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = (tail + " " + unit).strip() if len(tail) + 1 + len(unit) <= max_chars else unit
        else:
            current = (current + "\n" + unit).strip() if current else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_document(path: Path, root: Path, max_chars: int = 900, overlap: int = 150) -> List[Chunk]:
    source = path.relative_to(root).as_posix()
    chunks, position = [], 0
    for section in read_sections(path):
        for piece in split_text(section.text, max_chars, overlap):
            chunks.append(Chunk(source, section.heading, section.page, piece, position))
            position += 1
    return chunks


def iter_folder(root: Path, **kw) -> Iterable[Tuple[Path, List[Chunk]]]:
    for path in find_documents(root):
        yield path, chunk_document(path, root, **kw)
