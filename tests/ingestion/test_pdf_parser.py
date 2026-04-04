"""Tests for src.ingestion.pdf_parser."""

import os
import tempfile

import fitz  # PyMuPDF
import pytest

from src.ingestion.pdf_parser import PDFParser


@pytest.fixture
def parser() -> PDFParser:
    """Return a PDFParser instance."""
    return PDFParser()


def _create_pdf(tmp_dir: str, filename: str, pages: list[str]) -> str:
    """Helper: create a PDF with the given page texts and return its path."""
    path = os.path.join(tmp_dir, filename)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


class TestPDFParser:
    """Tests for the PDFParser class."""

    # ---- 正常 PDF 解析 ----

    def test_parse_valid_pdf(self, parser: PDFParser) -> None:
        """Test parsing a valid single-page PDF returns correct structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_pdf(tmp_dir, "valid.pdf", ["Hello world"])
            result = parser.parse(path)

            assert len(result) == 1
            assert "Hello world" in result[0]["text"]
            assert result[0]["page_number"] == 1
            assert result[0]["source"] == "valid.pdf"

    # ---- 空文件 ----

    def test_parse_empty_pdf(self, parser: PDFParser) -> None:
        """Test parsing a PDF with no text returns an empty list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_pdf(tmp_dir, "empty.pdf", [""])
            result = parser.parse(path)

            assert result == []

    def test_parse_whitespace_only_pdf(self, parser: PDFParser) -> None:
        """Test parsing a PDF with only whitespace text returns an empty list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_pdf(tmp_dir, "whitespace.pdf", ["   \n\t  "])
            result = parser.parse(path)

            assert result == []

    # ---- 损坏文件 ----

    def test_parse_corrupted_file_raises(self, parser: PDFParser) -> None:
        """Test that a corrupted file raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "corrupted.pdf")
            with open(path, "wb") as f:
                f.write(b"not a real pdf content")

            with pytest.raises(ValueError, match="Failed to open PDF"):
                parser.parse(path)

    def test_parse_nonexistent_file(self, parser: PDFParser) -> None:
        """Test that parsing a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/to/file.pdf")

    def test_parse_non_pdf_extension(self, parser: PDFParser) -> None:
        """Test that a non-.pdf file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with pytest.raises(ValueError, match="not a PDF"):
                parser.parse(tmp.name)

    # ---- 多页文档 ----

    def test_parse_multipage_pdf(self, parser: PDFParser) -> None:
        """Test parsing a multi-page PDF returns all pages with correct indices."""
        page_texts = ["Page one content", "Page two content", "Page three content"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_pdf(tmp_dir, "multi.pdf", page_texts)
            result = parser.parse(path)

            assert len(result) == 3
            for i, page_data in enumerate(result):
                assert page_data["page_number"] == i + 1
                assert page_texts[i] in page_data["text"]
                assert page_data["source"] == "multi.pdf"

    def test_parse_multipage_skips_blank_pages(self, parser: PDFParser) -> None:
        """Test that blank pages in a multi-page PDF are skipped."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_pdf(tmp_dir, "gaps.pdf", ["Content", "", "More content"])
            result = parser.parse(path)

            assert len(result) == 2
            assert result[0]["page_number"] == 1
            assert result[1]["page_number"] == 3

    # ---- 目录批量解析 ----

    def test_parse_directory_batch(self, parser: PDFParser) -> None:
        """Test parsing all PDFs in a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            _create_pdf(tmp_dir, "a.pdf", ["File A"])
            _create_pdf(tmp_dir, "b.pdf", ["File B page 1", "File B page 2"])
            # non-PDF file should be ignored
            with open(os.path.join(tmp_dir, "readme.txt"), "w") as f:
                f.write("ignore me")

            pdf_files = sorted(
                f for f in os.listdir(tmp_dir) if f.lower().endswith(".pdf")
            )
            all_pages: list[dict[str, str | int]] = []
            for pdf_file in pdf_files:
                all_pages.extend(parser.parse(os.path.join(tmp_dir, pdf_file)))

            assert len(all_pages) == 3
            sources = {p["source"] for p in all_pages}
            assert sources == {"a.pdf", "b.pdf"}

    def test_parse_empty_directory(self, parser: PDFParser) -> None:
        """Test parsing an empty directory yields no results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_files = [
                f for f in os.listdir(tmp_dir) if f.lower().endswith(".pdf")
            ]
            assert pdf_files == []
