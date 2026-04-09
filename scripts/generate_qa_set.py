"""Auto-generate an English QA evaluation test set from Danish PDF documents.

Calls an LLM (default: Qwen3-32B via Groq) to produce question / reference /
source-quote triples grounded in the source PDFs. The generated draft is
written to ``eval/qa_set_draft.yaml`` for human review and curation into the
final ``eval/qa_set.yaml``.

Each generated entry contains:
    question:           English question
    reference_en:       English reference answer (1–3 sentences)
    source_quote_da:    Verbatim Danish substring of the PDF page text
    source_doc:         PDF filename
    source_page_start:  First page of the section
    source_page_end:    Last page of the section
    category:           "fact" | "procedural" | "definition"
    quote_verified:     True if the Danish quote was found verbatim in the PDF
    reviewed:           Set to True manually after human review

Usage:
    python -m scripts.generate_qa_set [--max-sections-per-doc 3]
                                       [--questions-per-section 2]

Env vars (.env):
    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_...
    GROQ_MODEL=qwen/qwen3-32b
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings  # noqa: E402
from src.ingestion.pdf_parser import PDFParser  # noqa: E402
from src.ingestion.text_cleaner import TextCleaner  # noqa: E402
from src.provider import create_llm  # noqa: E402

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "eval", "qa_set_draft.yaml")

# Approximate character budget per section sent to the LLM. ~8K chars of
# Danish text is roughly 2.5K tokens, well within Qwen3-32B's 131K window
# even with a verbose system prompt.
_SECTION_CHAR_TARGET = 8000


_SYSTEM_PROMPT = """You are a question generator for a multilingual RAG \
evaluation test set.

You are given a section of a Danish university policy / regulation document. \
Your task is to generate factual question / answer pairs that test whether a \
retrieval system can find the right passage and answer the question correctly.

Strict rules:
1. Questions must be in ENGLISH (the test set targets English-speaking users \
querying Danish documents).
2. Each `reference_en` must be in ENGLISH and faithful to the source — do \
not add information not present in the text.
3. Each `source_quote_da` MUST be an EXACT verbatim substring of the source \
text I gave you. Do not paraphrase, do not translate, do not summarize. \
If you cannot find a clean verbatim quote that supports the answer, do \
not generate that question.
4. Quote length: keep `source_quote_da` between 30 and 400 characters — long \
enough to fully support the answer, short enough to be specific.
5. Categories: each question must be tagged with one of:
   - "fact": single specific fact (a number, a deadline, a definition, a rule)
   - "procedural": describes a process or sequence of steps
   - "definition": defines a term or concept
6. Avoid trivially generic questions like "What does the document say about \
X?" — questions should be answerable in 1-3 sentences with specific content.
7. Output STRICT JSON only, no markdown, no commentary, no thinking. Schema:

{
  "questions": [
    {
      "question": "string (English)",
      "reference_en": "string (English, 1-3 sentences)",
      "source_quote_da": "string (verbatim Danish substring of input)",
      "category": "fact" | "procedural" | "definition"
    }
  ]
}
"""


def _section_pages(
    pages: list[dict[str, str | int]], target_chars: int
) -> list[dict[str, Any]]:
    """Group consecutive pages into sections of approximately target_chars.

    Args:
        pages: List of page dicts from PDFParser.
        target_chars: Approximate character budget per section.

    Returns:
        List of section dicts with 'text', 'page_start', 'page_end'.
    """
    cleaner = TextCleaner()
    sections: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_pages: list[int] = []
    buf_chars = 0

    for page in pages:
        cleaned = cleaner.clean(str(page["text"]))
        cleaned = cleaner.remove_headers_footers(cleaned)
        if not cleaned.strip():
            continue
        page_no = int(page["page_number"])
        buf.append(cleaned)
        buf_pages.append(page_no)
        buf_chars += len(cleaned)
        if buf_chars >= target_chars:
            sections.append(
                {
                    "text": "\n\n".join(buf),
                    "page_start": buf_pages[0],
                    "page_end": buf_pages[-1],
                }
            )
            buf, buf_pages, buf_chars = [], [], 0

    if buf:
        sections.append(
            {
                "text": "\n\n".join(buf),
                "page_start": buf_pages[0],
                "page_end": buf_pages[-1],
            }
        )

    return sections


# Soft-hyphen line break: a hyphen at end of line followed by a lowercase
# letter is PDF reflow artefact (e.g. "dæknings-\nområde"), not a real hyphen.
_HYPHEN_LINEBREAK_RE = re.compile(r"-\s*\n\s*(?=[a-zæøå])")

# Quote and bullet glyphs whose presence differs between PDF text extraction
# and LLM-generated quotes. Stripping them entirely makes matching robust to
# straight-vs-curly quotes and to bullets that PyMuPDF drops on extraction.
_STRIP_CHARS = (
    "\u2018\u2019\u201a\u201b"  # single curly quotes
    "\u201c\u201d\u201e\u201f"  # double curly quotes
    "\u00ab\u00bb"               # « »
    "'\""                          # straight quotes
    "\u2022\u2023\u00b7\u25aa\u25e6\u25cf\u25cb"  # bullet glyphs
)
_STRIP_TRANSLATE = str.maketrans({c: "" for c in _STRIP_CHARS})

# Word/Wingdings list bullets land in the Unicode Private Use Area (e.g.
# U+F0B7) when extracted from PDFs. Drop the whole BMP PUA range.
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")


def _normalize_for_match(s: str) -> str:
    """Normalize text for tolerant verbatim-quote matching.

    Heals PDF-style soft hyphens at line breaks (e.g. ``dæknings-\\nområde``
    → ``dækningsområde``), removes quote characters whose straight/curly
    variants differ between PDF extraction and LLM output, drops bullet
    glyphs that PDF extraction tends to discard, and collapses whitespace
    runs to a single space.

    Args:
        s: Input string.

    Returns:
        Normalized string suitable for substring comparison.
    """
    s = _HYPHEN_LINEBREAK_RE.sub("", s)
    s = _PUA_RE.sub("", s)
    s = s.translate(_STRIP_TRANSLATE)
    return re.sub(r"\s+", " ", s).strip()


def _verify_quote(quote: str, source_text: str) -> bool:
    """Verify the quote is a verbatim substring of source_text after whitespace normalization.

    Args:
        quote: Candidate Danish quote produced by the LLM.
        source_text: Full source text the quote should originate from.

    Returns:
        True if the quote is found verbatim (modulo whitespace) in the source.
    """
    return _normalize_for_match(quote) in _normalize_for_match(source_text)


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, tolerating code fences and prose.

    Args:
        raw: Raw LLM output string.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON object could be extracted.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\nRaw (first 500 chars): {raw[:500]}"
        ) from exc


def _generate_for_section(
    llm: BaseChatModel,
    source_doc: str,
    section: dict[str, Any],
    questions_per_section: int,
) -> list[dict[str, Any]]:
    """Call the LLM to generate QA pairs for one document section.

    Args:
        llm: LangChain BaseChatModel instance.
        source_doc: Filename of the source document.
        section: Section dict with text, page_start, page_end.
        questions_per_section: Target number of questions to generate.

    Returns:
        List of QA dicts (may be empty if the LLM fails or no quotes verify).
    """
    user_prompt = (
        f"Generate exactly {questions_per_section} question/answer pairs from "
        f"the following Danish text. Remember: questions and reference answers "
        f"in ENGLISH, source_quote_da must be VERBATIM from the text below.\n\n"
        f"--- SOURCE TEXT ---\n{section['text']}\n--- END SOURCE TEXT ---"
    )
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(
            "LLM call failed for %s pages %d-%d: %s",
            source_doc,
            section["page_start"],
            section["page_end"],
            exc,
        )
        return []

    if not isinstance(raw, str):
        raw = str(raw)

    try:
        parsed = _parse_llm_json(raw)
    except ValueError as exc:
        logger.warning("Failed to parse LLM JSON for %s: %s", source_doc, exc)
        return []

    items = parsed.get("questions", [])
    if not isinstance(items, list):
        logger.warning("LLM 'questions' field is not a list for %s", source_doc)
        return []

    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        reference_en = item.get("reference_en")
        source_quote_da = item.get("source_quote_da")
        category = item.get("category", "fact")
        if not (
            isinstance(question, str)
            and isinstance(reference_en, str)
            and isinstance(source_quote_da, str)
        ):
            continue
        if not (question.strip() and reference_en.strip() and source_quote_da.strip()):
            continue
        verified = _verify_quote(source_quote_da, section["text"])
        result.append(
            {
                "question": question.strip(),
                "reference_en": reference_en.strip(),
                "source_quote_da": source_quote_da.strip(),
                "source_doc": source_doc,
                "source_page_start": section["page_start"],
                "source_page_end": section["page_end"],
                "category": category if category in {"fact", "procedural", "definition"} else "fact",
                "quote_verified": verified,
                "reviewed": False,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Auto-generate an English QA test set from Danish PDFs.",
    )
    parser.add_argument(
        "--max-sections-per-doc",
        type=int,
        default=3,
        help="Max sections to process per PDF (caps total questions; default 3).",
    )
    parser.add_argument(
        "--questions-per-section",
        type=int,
        default=2,
        help="Number of QA pairs to request per section (default 2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help=f"Output YAML path (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help="Process at most N PDFs (0 = all). Useful for smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the QA draft and write it to YAML."""
    args = parse_args()
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if settings.llm_provider != "groq":
        logger.warning(
            "LLM_PROVIDER is '%s', not 'groq'. The QA generator works with any "
            "provider but Qwen3-32B via Groq is recommended.",
            settings.llm_provider,
        )

    model_label = (
        settings.groq_model if settings.llm_provider == "groq" else settings.generation_model
    )
    logger.info("=== QA Draft Generation Start ===")
    logger.info("LLM provider: %s | model: %s", settings.llm_provider, model_label)

    llm = create_llm(settings)
    parser = PDFParser()

    pdf_files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    if args.limit_docs > 0:
        pdf_files = pdf_files[: args.limit_docs]
    logger.info("Found %d PDFs in %s", len(pdf_files), DOCS_DIR)

    all_questions: list[dict[str, Any]] = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DOCS_DIR, pdf_file)
        logger.info("Processing %s ...", pdf_file)
        try:
            pages = parser.parse(pdf_path)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pdf_file, exc)
            continue

        sections = _section_pages(pages, _SECTION_CHAR_TARGET)
        sections = sections[: args.max_sections_per_doc]
        logger.info("  -> %d sections", len(sections))

        for i, section in enumerate(sections, start=1):
            logger.info(
                "  Section %d/%d (pages %d-%d)",
                i,
                len(sections),
                section["page_start"],
                section["page_end"],
            )
            qa_items = _generate_for_section(
                llm=llm,
                source_doc=pdf_file,
                section=section,
                questions_per_section=args.questions_per_section,
            )
            verified = sum(1 for q in qa_items if q["quote_verified"])
            logger.info("    -> %d questions (%d verified)", len(qa_items), verified)
            all_questions.extend(qa_items)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generator": "scripts/generate_qa_set.py",
            "llm_provider": settings.llm_provider,
            "llm_model": model_label,
            "total_candidates": len(all_questions),
            "verified_quotes": sum(1 for q in all_questions if q["quote_verified"]),
        },
        "questions": all_questions,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            payload,
            fh,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        )

    logger.info("=== QA Draft Generation Complete ===")
    print(f"\nDraft written to: {out_path}")
    print(f"Total questions:  {len(all_questions)}")
    print(
        f"Verified quotes:  {sum(1 for q in all_questions if q['quote_verified'])} / {len(all_questions)}"
    )
    print("\nNext steps:")
    print("  1. Open the YAML and review each entry.")
    print("  2. Set `reviewed: true` on entries you want to keep.")
    print("  3. Edit any field that needs fixing.")
    print("  4. Save the curated set as eval/qa_set.yaml when done.")


if __name__ == "__main__":
    main()
