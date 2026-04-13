"""Snapshot tests ensuring YAML prompts render byte-identical to the legacy inline strings."""

from src.agent.prompts import registry as _reg
from src.agent.prompts import get_prompt, render_prompt


def setup_module(module) -> None:  # noqa: ANN001
    """Force a fresh registry load before snapshot tests."""
    _reg.reload()


# Golden literals — frozen copies of the pre-migration strings. Any future
# change to the YAML must preserve these byte-for-byte. Do NOT replace these
# with imports from the source modules; that would make the test tautological
# after the source modules are migrated to read from the registry.

_GOLDEN_INTENT_CLASSIFY = (
    "You are an intent classifier. Given a user query, classify it into exactly "
    "one of the following categories: factual, summary, comparison, procedural, unknown.\n\n"
    "- factual: the user asks for a specific fact or piece of information.\n"
    "- summary: the user wants a summary or overview of a topic.\n"
    "- comparison: the user wants to compare two or more things.\n"
    "- procedural: the user asks how to do something step by step.\n"
    "- unknown: the query does not fit any of the above.\n\n"
    "Respond with ONLY the category name in lowercase, nothing else."
)

_GOLDEN_PLANNER = (
    "You are a planning assistant for the University of Copenhagen (KU) document system.\n\n"
    "Given a user question, produce a JSON list of 1–4 steps needed to answer it.\n"
    "Each step is an object with:\n"
    '  - "action": one of "search", "search_within", "multi_search", '
    '"summarize", "list_docs", "fetch_doc"\n'
    '  - "detail": a short description of what to do (e.g. the search query, document ID)\n\n'
    "Rules:\n"
    "- IMPORTANT: Most questions probably only need 1 step. Only use 2+ steps when the question explicitly asks about multiple distinct topics.\n"
    "- For simple factual questions: 1 search step is enough.\n"
    "- For comparison questions: use multi_search or separate search steps.\n"
    "- For document overview requests: use summarize.\n"
    "- For questions with multiple aspects: use 2–4 separate steps.\n"
    "- Always end with the steps needed; do NOT include a final 'answer' step.\n\n"
    "Reply with ONLY the JSON array, nothing else. No explanation, no thinking.\n\n"
    "Examples:\n"
    'Question: "What is the exam policy?"\n'
    '[{"action": "search", "detail": "KU eksamensregler"}]\n\n'
    'Question: "Compare vacation rules for academic vs administrative staff"\n'
    '[{"action": "search", "detail": "ferieregler videnskabeligt personale"}, '
    '{"action": "search", "detail": "ferieregler administrativt personale"}]\n\n'
    'Question: "Summarize the AI policy document"\n'
    '[{"action": "summarize", "detail": "ku_ai_policy.pdf"}]\n\n'
    'Question: "Which documents are about AI? Summarize and find the rules for written exams"\n'
    '[{"action": "list_docs", "detail": "list all available documents"}, '
    '{"action": "search", "detail": "AI dokumenter KU"}, '
    '{"action": "search", "detail": "regler skriftlige opgaver eksamen GAI"}]\n\n'
    "Now plan for this question:\n"
)

_GOLDEN_EXECUTOR_SYSTEM = (
    "/no_think\n"
    "You are executing ONE step of a plan to answer a user's question about "
    "University of Copenhagen (KU) documents.\n\n"
    "You have retrieval tools available. Execute the step described below, "
    "then summarise what you found in 2-3 sentences. If you find nothing "
    "relevant, say so clearly.\n\n"
    "Do NOT produce a final answer — just report what you found for this step."
)

_GOLDEN_SYNTHESIZER = (
    "You are a helpful assistant for administrative staff at the University "
    "of Copenhagen (KU).\n\n"
    "Below are the results gathered from multiple research steps. "
    "Synthesize them into a single coherent answer to the user's original question.\n\n"
    "Guidelines:\n"
    "- Cite document sources using [1], [2], etc.\n"
    "- Answer in the same language as the user's question.\n"
    "- Be concise but thorough.\n"
    "- If some steps found no results, acknowledge gaps honestly.\n\n"
)


def test_intent_classify_matches_golden() -> None:
    assert get_prompt("intent_classify").template == _GOLDEN_INTENT_CLASSIFY


def test_planner_matches_golden() -> None:
    assert get_prompt("planner").template == _GOLDEN_PLANNER


def test_executor_system_matches_golden() -> None:
    assert get_prompt("executor_system").template == _GOLDEN_EXECUTOR_SYSTEM


def test_synthesizer_matches_golden() -> None:
    assert get_prompt("synthesizer").template == _GOLDEN_SYNTHESIZER


def test_detect_language_and_intent_renders_identically() -> None:
    valid_intents = "factual, summary, comparison, procedural, unknown"
    query = "Hvor mange feriedage har jeg?"
    expected = (
        "You are given a user query. Do TWO things:\n"
        "1. Detect the language of the query (reply with the language name in English, "
        "e.g. 'Danish', 'English', 'German', 'Chinese', 'Japanese').\n"
        "2. Classify the intent into exactly one of: "
        f"{valid_intents}.\n\n"
        "Reply with EXACTLY two lines, nothing else:\n"
        "language: <language>\n"
        "intent: <intent>\n\n"
        f"Query: {query}"
    )
    rendered = render_prompt(
        "detect_language_and_intent", valid_intents=valid_intents, query=query
    )
    assert rendered == expected


def test_translate_query_renders_identically() -> None:
    target = "Danish"
    query = "How many vacation days do I have?"
    expected = (
        f"Translate the following text to {target}. "
        "Reply with ONLY the translated text, nothing else.\n\n"
        f"Text: {query}"
    )
    assert render_prompt("translate_query", target=target, query=query) == expected


def test_broaden_query_renders_identically() -> None:
    query = "exam rules"
    retrieval_query = "eksamensregler"
    expected = (
        "The following search query did not return good results from "
        "the document database. Rewrite it to be broader or use "
        "different keywords while keeping the same meaning. "
        "Reply with ONLY the rewritten query, nothing else.\n\n"
        f"Original question: {query}\n"
        f"Failed search query: {retrieval_query}"
    )
    assert render_prompt("broaden_query", query=query, retrieval_query=retrieval_query) == expected


def test_detect_languages_renders_identically() -> None:
    sample_text = "Dette er en test.\n---\nThis is a test."
    expected = (
        "You are a language detector. The text samples below come from "
        "different documents in a knowledge base. Identify ALL distinct "
        "languages present across the samples (do not list a language more "
        "than once). Reply with ONLY the language names in English, one per "
        "line, no explanation.\n\n"
        f"Samples:\n{sample_text}"
    )
    assert render_prompt("detect_languages", sample_text=sample_text) == expected


def test_multi_query_decompose_renders_identically() -> None:
    lang_clause = "The queries should be in Danish (the document base is Danish)."
    question = "Compare rules between master and bachelor exams."
    expected = (
        "You are a search query planner. Given a complex question, "
        "decompose it into 2-4 simple, independent search queries that "
        f"together cover all aspects of the question. {lang_clause}\n\n"
        "Reply with ONLY the queries, one per line, nothing else.\n\n"
        f"Question: {question}"
    )
    rendered = render_prompt(
        "multi_query_decompose", lang_clause=lang_clause, question=question
    )
    assert rendered == expected


def test_summarize_document_renders_identically() -> None:
    document_id = "ku_ai_policy.pdf"
    full_text = "Section 1.\n\nSection 2."
    expected = (
        "Produce a structured summary of the following document. "
        "Include:\n"
        "1. Document title/topic\n"
        "2. Key points (3-7 bullet points)\n"
        "3. Important rules, deadlines, or requirements mentioned\n"
        "4. Who the document applies to\n\n"
        "Write the summary in the same language as the document.\n\n"
        f"Document ID: {document_id}\n\n"
        f"Document text:\n{full_text}"
    )
    assert render_prompt(
        "summarize_document", document_id=document_id, full_text=full_text
    ) == expected


def test_registry_raises_on_unknown_prompt() -> None:
    import pytest
    with pytest.raises(KeyError):
        get_prompt("does_not_exist")


def test_registry_raises_on_unknown_version() -> None:
    import pytest
    with pytest.raises(KeyError):
        get_prompt("intent_classify", version="v999")
