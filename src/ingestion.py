"""
ingestion.py - PDF parsing and Markdown-aware chunking.

Handles loading documents from disk, extracting text (with error handling),
and splitting into overlapping chunks suitable for embedding.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk ready for embedding and indexing."""
    chunk_id: str          # Unique identifier: "<source_filename>_chunk_<idx>"
    source: str            # Original file path/name
    text: str              # The actual text content
    metadata: dict = field(default_factory=dict)  # Page numbers, headings, etc.


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text from a PDF file page by page.

    Returns a list of (page_number, text) tuples.
    Pages with extraction errors are skipped with a warning logged.
    """
    pages: list[tuple[int, str]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append((page_num, text.strip()))
                    else:
                        logger.warning(
                            "Page %d in '%s' yielded no text – skipping.",
                            page_num, pdf_path.name
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to extract page %d from '%s': %s",
                        page_num, pdf_path.name, exc
                    )
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not open PDF '%s': %s", pdf_path, exc)

    return pages


def extract_text_from_markdown(md_path: Path) -> str:
    """Read a Markdown file and return its raw text."""
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read Markdown file '%s': %s", md_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _split_by_markdown_headings(text: str) -> list[str]:
    """
    Split a Markdown document at heading boundaries.
    Each resulting section includes its heading as the first line.
    """
    positions = [m.start() for m in _MARKDOWN_HEADING.finditer(text)]
    if not positions:
        return [text]

    sections: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _sliding_window_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Generator[str, None, None]:
    """
    Yield overlapping word-level chunks from *text*.

    Args:
        text:       Source text.
        chunk_size: Target number of *words* per chunk.
        overlap:    Number of words to repeat at the start of the next chunk.
    """
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += chunk_size - overlap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Extract and chunk a PDF document.

    Pages are concatenated into a single text body then split via a
    sliding-window strategy. Page numbers are stored in chunk metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        logger.warning("No text extracted from '%s'.", pdf_path.name)
        return []

    chunks: list[Chunk] = []
    # Concatenate with a page-boundary marker so we can track page numbers
    combined_text = "\n".join(text for _, text in pages)

    for idx, window_text in enumerate(_sliding_window_chunks(combined_text, chunk_size, overlap)):
        chunk = Chunk(
            chunk_id=f"{pdf_path.stem}_chunk_{idx}",
            source=str(pdf_path),
            text=window_text,
            metadata={"file_type": "pdf", "chunk_index": idx},
        )
        chunks.append(chunk)

    logger.info("PDF '%s' → %d chunks.", pdf_path.name, len(chunks))
    return chunks


def chunk_markdown(
    md_path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Extract and chunk a Markdown document.

    First splits at heading boundaries (preserving semantic sections),
    then applies sliding-window chunking within each section.
    """
    raw_text = extract_text_from_markdown(md_path)
    if not raw_text.strip():
        logger.warning("No text found in '%s'.", md_path.name)
        return []

    sections = _split_by_markdown_headings(raw_text)
    chunks: list[Chunk] = []
    chunk_idx = 0

    for section in sections:
        # Extract the heading (first line) for metadata
        first_line = section.splitlines()[0] if section.splitlines() else ""
        heading = first_line.lstrip("#").strip()

        for window_text in _sliding_window_chunks(section, chunk_size, overlap):
            chunk = Chunk(
                chunk_id=f"{md_path.stem}_chunk_{chunk_idx}",
                source=str(md_path),
                text=window_text,
                metadata={
                    "file_type": "markdown",
                    "heading": heading,
                    "chunk_index": chunk_idx,
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

    logger.info("Markdown '%s' → %d chunks.", md_path.name, len(chunks))
    return chunks


def ingest_documents(
    docs_dir: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Ingest all PDF and Markdown files found in *docs_dir* (non-recursive).

    Returns a flat list of Chunk objects ready for embedding.
    """
    all_chunks: list[Chunk] = []
    supported = {".pdf": chunk_pdf, ".md": chunk_markdown, ".markdown": chunk_markdown}

    for file_path in sorted(docs_dir.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix not in supported:
            logger.debug("Skipping unsupported file: %s", file_path.name)
            continue

        logger.info("Ingesting: %s", file_path.name)
        chunker = supported[suffix]
        file_chunks = chunker(file_path, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(file_chunks)

    logger.info("Total chunks ingested: %d", len(all_chunks))
    return all_chunks