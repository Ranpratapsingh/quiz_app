"""
tests/test_ingestion.py
-----------------------
Unit tests for the document ingestion module.
Run with: pytest tests/
"""

import pytest
import os
import tempfile
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ingestion import (
    clean_text,
    split_into_chunks,
    extract_txt,
    extract_csv,
    ingest_document,
    ingest_text,
)


# ── clean_text ─────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_excessive_whitespace(self):
        result = clean_text("Hello   world")
        assert "  " not in result

    def test_collapses_triple_newlines(self):
        result = clean_text("Para 1\n\n\n\n\nPara 2")
        assert "\n\n\n" not in result

    def test_removes_pure_number_lines(self):
        text = "Important content\n\n42\n\nMore content"
        result = clean_text(text)
        assert "\n42\n" not in result or "42" not in result.splitlines()

    def test_preserves_actual_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = clean_text(text)
        assert "quick brown fox" in result


# ── split_into_chunks ──────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_returns_list(self):
        chunks = split_into_chunks("Hello world. " * 100)
        assert isinstance(chunks, list)

    def test_chunks_not_empty(self):
        text = "This is a test paragraph.\n\nAnd another one here.\n\nThird paragraph."
        chunks = split_into_chunks(text)
        assert all(len(c) > 0 for c in chunks)

    def test_respects_chunk_size(self):
        text = " ".join(["word"] * 1000)
        chunks = split_into_chunks(text, chunk_size=200)
        # Allow 1.5x for the long-paragraph fallback path
        for c in chunks:
            assert len(c) <= 200 * 2

    def test_short_text_gives_one_chunk(self):
        text = "This is a short document with one paragraph only."
        chunks = split_into_chunks(text, chunk_size=800)
        assert len(chunks) == 1

    def test_drops_tiny_fragments(self):
        text = "A\n\nThis is a real paragraph with enough content to survive.\n\nB"
        chunks = split_into_chunks(text, chunk_size=800)
        # Single-char fragments like "A" and "B" should be dropped
        for c in chunks:
            assert len(c) > 50


# ── extract_txt ────────────────────────────────────────────────────────────────

class TestExtractTxt:
    def test_basic_utf8(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = extract_txt(str(f))
        assert result == "Hello, world!"

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9".encode("latin-1"))
        result = extract_txt(str(f))
        assert len(result) > 0


# ── extract_csv ────────────────────────────────────────────────────────────────

class TestExtractCsv:
    def test_basic_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        result = extract_csv(str(f))
        assert "name" in result.lower()
        assert "alice" in result.lower()
        assert "Row 1" in result

    def test_reports_column_count(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n")
        result = extract_csv(str(f))
        assert "a, b, c" in result or "columns: a" in result.lower()

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("col1,col2\n")
        result = extract_csv(str(f))
        assert result == ""


# ── ingest_document ────────────────────────────────────────────────────────────

class TestIngestDocument:
    def test_unsupported_type_raises(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            ingest_document(str(f))

    def test_empty_txt_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n   \n   ")
        with pytest.raises(RuntimeError):
            ingest_document(str(f))

    def test_valid_txt_returns_chunks(self, tmp_path):
        f = tmp_path / "doc.txt"
        content = "\n\n".join([
            "This is the first paragraph with plenty of words to form a real chunk.",
            "This is the second paragraph with its own unique content and information.",
            "The third paragraph adds more detail and context to the document.",
        ])
        f.write_text(content)
        chunks = ingest_document(str(f))
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)


class TestIngestText:
    def test_valid_text_returns_chunks(self):
        text = (
            "Artificial intelligence helps computers learn from data and make decisions. "
            "Machine learning is a subset of AI that focuses on pattern recognition.\n\n"
            "Deep learning uses layered neural networks to solve more complex tasks."
        )
        chunks = ingest_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_empty_text_raises(self):
        with pytest.raises(RuntimeError):
            ingest_text("   \n   ")
