from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import pypdf
from docx import Document


@dataclass
class Chunk:
    text: str
    source: str          # filename
    page: int            # 0-indexed; -1 for table chunks
    chunk_type: str      # "prose" | "table"
    chunk_id: int = 0    # assigned after collection


def _chunks_from_text(text: str, source: str, page: int, chunk_type: str,
                      chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    """Split a long text into overlapping chunks."""
    words = text.split()
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(words)), step):
        chunk_text = " ".join(words[i: i + chunk_size]).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source=source, page=page, chunk_type=chunk_type))
    return chunks


def ingest_pdf(path: Path, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    chunks: List[Chunk] = []
    reader = pypdf.PdfReader(str(path))
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Skip pages with corrupt inline images or unreadable streams
            continue
        if text.strip():
            chunks.extend(_chunks_from_text(text, path.name, page_num, "prose", chunk_size, overlap))
    return chunks


def ingest_docx(path: Path, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    doc = Document(str(path))
    chunks: List[Chunk] = []

    # Prose: paragraphs
    prose_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if prose_text.strip():
        chunks.extend(_chunks_from_text(prose_text, path.name, 0, "prose", chunk_size, overlap))

    # Tables: each table row becomes a separate chunk (preserves structure)
    for table_idx, table in enumerate(doc.tables):
        headers = [cell.text.strip() for cell in table.rows[0].cells] if table.rows else []
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(
                f"{h}: {c}" for h, c in zip(headers, cells) if c
            )
            if row_text.strip():
                chunks.append(Chunk(
                    text=row_text,
                    source=path.name,
                    page=-1,
                    chunk_type="table",
                ))
    return chunks


def ingest_tabular(path: Path) -> List[Chunk]:
    """Ingest CSV or Excel files. Each row → one chunk."""
    chunks: List[Chunk] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(str(path))
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(str(path))
    else:
        return chunks

    headers = df.columns.tolist()
    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {val}" for col, val in zip(headers, row) if str(val).strip())
        if row_text.strip():
            chunks.append(Chunk(text=row_text, source=path.name, page=-1, chunk_type="table"))
    return chunks


def ingest_file(path: Path, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, chunk_size, overlap)
    elif suffix == ".docx":
        return ingest_docx(path, chunk_size, overlap)
    elif suffix in (".csv", ".xlsx", ".xls"):
        return ingest_tabular(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def ingest_directory(directory: Path, chunk_size: int = 500, overlap: int = 100) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    supported = {".pdf", ".docx", ".csv", ".xlsx", ".xls"}
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in supported:
            try:
                file_chunks = ingest_file(file_path, chunk_size, overlap)
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"[ingest] Skipping {file_path.name}: {e}")

    # Assign global chunk IDs
    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_id = idx

    return all_chunks
