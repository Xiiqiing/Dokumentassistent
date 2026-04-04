"""PDF parsing module using PyMuPDF (fitz)."""

import logging
import os

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParser:
    """Parses PDF files and extracts raw text with metadata."""

    def parse(self, file_path: str) -> list[dict[str, str | int]]:
        """Extract text and metadata from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of dicts, each containing 'text', 'page_number',
            and 'source' keys.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {file_path}")

        logger.info("Parsing PDF: %s", file_path)
        source = os.path.basename(file_path)
        pages: list[dict[str, str | int]] = []

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise ValueError(f"Failed to open PDF: {file_path}") from exc

        try:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages.append({
                        "text": text,
                        "page_number": page_num,
                        "source": source,
                    })
        finally:
            doc.close()

        logger.info("Extracted %d pages from %s", len(pages), source)
        return pages
