"""Load the markdown knowledge base and split it into retrievable chunks.

Chunking is by markdown section (`## heading`): maintenance docs are already
organized into self-contained sections ("Spindle overheating", "Loss of pressing
force", ...), which are the natural retrieval unit -- each answers one kind of
question. Each chunk keeps its document title and section heading both as
metadata and prepended to the text, so the embedder sees the topic words and a
stable, human-readable chunk_id (`<file>::<heading>`) can be cited and used as
the evaluation ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import CORPUS_DIR


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "<file-stem>::<section heading>" -- stable, used by the eval set
    doc_title: str
    heading: str
    text: str          # doc title + heading + body, for embedding and display
    source_file: str


def _split_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    lines = md.splitlines()
    doc_title = ""
    if lines and lines[0].startswith("# "):
        doc_title = lines[0][2:].strip()
        lines = lines[1:]
    sections: list[tuple[str, str]] = []
    heading, body = None, []
    for ln in lines:
        m = re.match(r"^##\s+(.*)", ln)
        if m:
            if heading is not None:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = m.group(1).strip(), []
        else:
            body.append(ln)
    if heading is not None:
        sections.append((heading, "\n".join(body).strip()))
    return doc_title, sections


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc_title, sections = _split_sections(path.read_text(encoding="utf-8"))
        for heading, body in sections:
            if not body:
                continue
            chunks.append(Chunk(
                chunk_id=f"{path.stem}::{heading}",
                doc_title=doc_title,
                heading=heading,
                text=f"{doc_title} — {heading}\n{body}",
                source_file=path.name,
            ))
    if not chunks:
        raise FileNotFoundError(f"No markdown chunks found under {corpus_dir}")
    return chunks
