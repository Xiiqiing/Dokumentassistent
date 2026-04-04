"""Text cleaning and normalization for parsed PDF content."""

import logging
import re

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and normalizes raw text extracted from PDFs."""

    def clean(self, raw_text: str) -> str:
        """Clean raw text by removing artifacts and normalizing whitespace.

        Args:
            raw_text: The raw text extracted from a PDF page.

        Returns:
            Cleaned and normalized text string.
        """
        text = raw_text
        # Remove null bytes and control characters (keep newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize unicode whitespace to regular spaces
        text = re.sub(r"\u00a0", " ", text)
        # Remove soft hyphens
        text = text.replace("\u00ad", "")
        # Collapse multiple spaces into one
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse three or more newlines into two
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        text = "\n".join(line.strip() for line in text.splitlines())
        # Strip leading/trailing whitespace overall
        text = text.strip()
        return text

    def remove_headers_footers(self, text: str) -> str:
        """Remove repeating headers and footers from text.

        Args:
            text: Text that may contain headers/footers.

        Returns:
            Text with headers and footers removed.
        """
        lines = text.splitlines()
        if len(lines) < 3:
            return text
        # Remove common page-number-only lines (e.g. "  3  ", "- 12 -", "Side 5")
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Skip standalone page numbers
            if re.match(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$", stripped):
                continue
            # Skip lines like "Side 3" or "Page 3" (Danish/English)
            if re.match(r"^(side|page)\s+\d{1,4}$", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
