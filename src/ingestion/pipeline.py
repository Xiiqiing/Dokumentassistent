"""End-to-end ingestion pipeline: parse, clean, chunk, and store."""

import logging
import os

from langchain_core.embeddings import Embeddings

from src.models import ChunkStrategy, DocumentChunk
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_cleaner import TextCleaner
from src.ingestion.chunker import create_chunker

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full document ingestion workflow."""

    def __init__(
        self,
        strategy: ChunkStrategy,
        chunk_size: int,
        chunk_overlap: int,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            strategy: Chunking strategy to apply.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            embeddings: LangChain Embeddings instance (required for SEMANTIC strategy).
        """
        self.parser = PDFParser()
        self.cleaner = TextCleaner()
        self.chunker = create_chunker(strategy, chunk_size, chunk_overlap, embeddings)

    def ingest_text_file(self, file_path: str) -> list[DocumentChunk]:
        """Ingest a plain text file through the cleaning and chunking pipeline.

        Args:
            file_path: Path to the text file.

        Returns:
            List of processed DocumentChunk objects ready for indexing.
        """
        logger.info("Ingesting text file: %s", file_path)
        document_id = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()

        cleaned = self.cleaner.clean(raw_text)
        if not cleaned.strip():
            logger.warning("Text file produced no content: %s", file_path)
            return []

        metadata: dict[str, str | int] = {
            "source": document_id,
            "page_number": 1,
        }
        chunks = self.chunker.chunk(cleaned, document_id, metadata)
        logger.info("Ingested %d chunks from %s", len(chunks), file_path)
        return chunks

    def ingest_pdf(self, file_path: str) -> list[DocumentChunk]:
        """Ingest a single PDF file through the full pipeline.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of processed DocumentChunk objects ready for indexing.
        """
        logger.info("Ingesting PDF: %s", file_path)
        document_id = os.path.basename(file_path)

        pages = self.parser.parse(file_path)
        all_chunks: list[DocumentChunk] = []

        for page in pages:
            raw_text = str(page["text"])
            cleaned = self.cleaner.clean(raw_text)
            cleaned = self.cleaner.remove_headers_footers(cleaned)
            if not cleaned.strip():
                continue
            metadata: dict[str, str | int] = {
                "source": str(page["source"]),
                "page_number": int(page["page_number"]),
            }
            chunks = self.chunker.chunk(cleaned, document_id, metadata)
            all_chunks.extend(chunks)

        logger.info("Ingested %d chunks from %s", len(all_chunks), file_path)
        return all_chunks

    _PDF_EXTENSIONS: set[str] = {".pdf"}
    _SKIP_EXTENSIONS: set[str] = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
        ".zip", ".gz", ".tar", ".rar", ".7z",
        ".exe", ".dll", ".so", ".dylib",
        ".mp3", ".mp4", ".wav", ".avi",
        ".pyc", ".pyo", ".class",
    }

    def ingest_directory(self, directory_path: str) -> list[DocumentChunk]:
        """Ingest all supported files in a directory.

        PDF files are parsed with PyMuPDF; all other non-binary files
        are read as plain text.

        Args:
            directory_path: Path to the directory containing documents.

        Returns:
            List of all processed DocumentChunk objects.
        """
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_files = sorted(os.listdir(directory_path))
        pdf_files = [f for f in all_files if os.path.splitext(f)[1].lower() in self._PDF_EXTENSIONS]
        text_files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() not in self._PDF_EXTENSIONS | self._SKIP_EXTENSIONS
            and os.path.isfile(os.path.join(directory_path, f))
        ]

        logger.info(
            "Found %d PDF files and %d text files in %s",
            len(pdf_files), len(text_files), directory_path,
        )

        all_chunks: list[DocumentChunk] = []

        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            all_chunks.extend(self.ingest_pdf(file_path))

        for text_file in text_files:
            file_path = os.path.join(directory_path, text_file)
            try:
                all_chunks.extend(self.ingest_text_file(file_path))
            except UnicodeDecodeError:
                logger.warning("Skipping binary file: %s", file_path)

        logger.info("Total chunks from directory: %d", len(all_chunks))
        return all_chunks
