"""Load policy documents and split them into heading-aware chunks.

Documents are Markdown (or text) with YAML front matter:

    ---
    doc_id: hr-leave-001
    title: Annual Leave Policy
    allowed_roles: [employee]
    version: "2026-01"
    owner: People Team
    ---

PDFs are supported with the `pdf` extra and a sidecar `<file>.meta.yaml` holding the same keys.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    text: str
    source: str
    allowed_roles: list[str]
    version: str | None = None
    owner: str | None = None
    classification: str = "internal"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            [self.text, self.title, sorted(self.allowed_roles), self.version, self.classification], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Chunk:
    doc_id: str
    index: int
    heading: str
    text: str

    def embedding_text(self, title: str) -> str:
        # A contextual header (title and section path) keeps short chunks retrievable.
        return f"{title} › {self.heading}\n\n{self.text}"


def load_document(path: Path) -> SourceDocument:
    if path.suffix.lower() == ".pdf":
        meta = yaml.safe_load(path.with_suffix(".meta.yaml").read_text())
        text = _read_pdf(path)
    else:
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(raw)
        meta = yaml.safe_load(match.group(1)) if match else {}
        text = raw[match.end() :] if match else raw
    missing = [k for k in ("doc_id", "title") if not meta.get(k)]
    if missing:
        raise ValueError(f"{path}: front matter is missing {', '.join(missing)}")
    roles = meta.get("allowed_roles") or ["employee"]
    return SourceDocument(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        text=text.strip(),
        source=str(path),
        allowed_roles=[str(r) for r in roles],
        version=str(meta["version"]) if meta.get("version") else None,
        owner=meta.get("owner"),
        classification=meta.get("classification", "internal"),
        metadata={k: v for k, v in meta.items() if k not in {"doc_id", "title", "allowed_roles", "version"}},
    )


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("PDF ingestion needs the 'pdf' extra: pip install 'company-assistant[pdf]'") from e
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _sections(doc: SourceDocument) -> list[tuple[str, str]]:
    """Split on headings; each section carries its breadcrumb (e.g. 'Annual leave › Carry-over')."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [(doc.title, [])]
    for line in doc.text.splitlines():
        m = _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
            crumbs = [t for _, t in stack if t != doc.title] or [doc.title]
            sections.append((" › ".join(crumbs), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    parts, current = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    # A single sentence longer than max_chars is hard-split.
    return [p[i : i + max_chars] for p in parts for i in range(0, len(p), max_chars)]


def chunk_document(doc: SourceDocument, max_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _sections(doc):
        paragraphs = [p for raw in re.split(r"\n\s*\n", body) for p in _split_long(raw.strip(), max_chars) if p]
        current: list[str] = []
        for para in paragraphs:
            if current and len("\n\n".join(current + [para])) > max_chars:
                chunks.append(Chunk(doc.doc_id, len(chunks), heading, "\n\n".join(current)))
                tail = current[-1]
                current = [tail] if len(tail) <= overlap_chars else []
            current.append(para)
        if current:
            chunks.append(Chunk(doc.doc_id, len(chunks), heading, "\n\n".join(current)))
    return chunks
