"""Streamlit frontend for Dokumentintelligens-system.

Calls the FastAPI backend at http://localhost:8000.
Single-page document search interface with clean sans-serif design.
"""

import html
import json
import os
import random
import uuid

import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Per-browser session ID (persisted via cookie, falls back to session_state)
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Example questions — drawn from the documents in docs/
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS: list[str] = [
    "Hvad er reglerne for brug af generativ AI til eksamen på KU?",
    "Hvordan håndteres uansøgt afsked begrundet i institutionens forhold?",
    "Hvad er de disciplinære foranstaltninger over for studerende?",
    "Hvordan skal klager over medarbejdere og ledere behandles?",
    "Hvad er retningslinjerne for afholdelse af MUS-samtaler?",
    "Hvordan er års- og skemastrukturen organiseret på KU?",
    "Hvilke regler gælder for eksamenstilmelding og afmelding?",
    "Hvordan skal studerende dokumentere brug af GAI i skriftlige opgaver?",
    "Hvad er kommunernes ansvar ved brug af generativ AI?",
    "Hvilke principper gælder for akademisk integritet ved brug af AI?",
    "Hvornår kan en leder afvise en klage som åbenbart grundløs?",
    "Hvad er reglerne for forlænget tid til eksamen?",
]

# ---------------------------------------------------------------------------
# Internationalisation — all UI strings live here
# ---------------------------------------------------------------------------
TEXTS: dict[str, dict[str, str]] = {
    "da": {
        "page_title": "Dokumentintelligens-system",
        "lang_label": "Sprog",
        "sidebar_heading": "Om systemet",
        "sidebar_body": (
            "- **Python + FastAPI** REST-backend\n"
            "- **Ustruktureret data** — File-parsing, preprocessing, "
            "tre chunking-strategier\n"
            "- **Embedding-modeller** — flersproget semantisk "
            "vektorrepræsentation\n"
            "- **Vektordatabase + hybrid søgning** — Qdrant (semantisk) "
            "+ BM25 (leksikalsk)\n"
            "- **Reranking** — cross-encoder for præcis relevans\n"
            "- **RAG-arkitektur** — LangChain + LangGraph-orkestreret pipeline\n"
            "- **LLM-integration** — provider-agnostisk, prompt-styret "
            "svargenerering\n"
            "- **Evaluering** — RAGAS-baseret kvalitetsmåling\n"
            "- **Agent Flows** — LangGraph Plan-and-Execute med værktøjskald og samtalehukommelse\n"
            "- [**Kildedokumenter**](https://github.com/Xiiqiing/Dokumentassistent/tree/main/docs)"
            " — de dokumenter systemet er indekseret fra"
        ),
        "chunking_label": "Chunking-strategi",
        "chunking_help": "Vælg hvordan dokumenterne opdeles i tekststykker.",
        "topk_label": "Antal kilder (top_k)",
        "topk_help": "Antal dokumentfragmenter der hentes fra søgeindekset.",
        "title": "Dokumentintelligens-system",
        "title_badge": "",
        "subtitle": (
            "Et dokumentintelligens-system bygget på en RAG-arkitektur, dækkende file-indlæsning, semantisk chunking, "
            "hybrid søgning med reranking "
            "og LLM-genererede svar med kildehenvisninger. LLM-laget er provider-agnostisk. "
            "To tilstande: en LangGraph Plan-and-Execute-agent (standard) med samtalehukommelse til komplekse forespørgsler, "
            "og en foruddefineret pipeline til lette modeller. Søgekvaliteten evalueres med RAGAS."
        ),
        "search_label": "Stil et spørgsmål om ... ",
        "search_placeholder": "F.eks.: Hvad er reglerne for behandling af personoplysninger?",
        "search_button": "Søg",
        "example_button": "Tilfældigt eksempel",
        "spinner": "Søger i dokumenterne ...",
        "status_label": "Tænker",
        "status_done": "Færdig",
        "status_error": "Noget gik galt",
        "confidence_label": "Konfidensgrad",
        "intent_label": "Intent",
        "strategy_label": "Strategi",
        "no_answer": "Intet svar modtaget.",
        "sources_label": "Kilder",
        "page_label": "side",
        "no_sources": "Ingen kilder fundet for denne forespørgsel.",
        "empty_warning": "Indtast venligst et spørgsmål.",
        "err_connection": (
            "Kunne ikke oprette forbindelse til API-serveren. "
            "Kontroller at backend kører på http://localhost:8000."
        ),
        "err_api": "API-fejl",
        "err_rate_limit": "For mange samtidige forespørgsler, eller API-kvoten er midlertidigt opbrugt. Vent venligst et øjeblik, og prøv igen.",
        "err_timeout": "Forespørgslen tog for lang tid. Prøv igen.",
        "unknown": "ukendt",
        "model_heading": "Aktuel model",
        "model_llm": "LLM",
        "model_embedding": "Embedding",
        "model_unavailable": "Kunne ikke hente modelinfo.",
        "pipeline_heading": "Pipeline-detaljer",
        "pipeline_translation": "Oversættelse",
        "pipeline_original": "Original forespørgsel",
        "pipeline_translated": "Oversat til dansk",
        "pipeline_lang": "Sprog registreret",
        "pipeline_no_translation": "Ingen oversættelse nødvendig",
        "pipeline_bm25": "BM25-resultater (leksikalsk søgning)",
        "pipeline_dense": "Vektorsøgning (semantisk)",
        "pipeline_fused": "RRF-fusioneret rækkefølge",
        "pipeline_reranked": "Reranking (endelig rækkefølge)",
        "pipeline_doc": "Dokument",
        "pipeline_score": "Score",
        "pipeline_rank": "#",
        "pipeline_no_results": "Ingen resultater",
        "pipeline_score_change": "Score-ændring",
        "pipeline_plan_steps": "Udførelsesplan",
        "pipeline_tool_calls": "Værktøjskald",
        "synthesize_status": "Syntetiserer endeligt svar ...",
        "example_note": "",
    },
    "en": {
        "page_title": "Document Intelligence System",
        "lang_label": "Language",
        "sidebar_heading": "About the system",
        "sidebar_body": (
            "- **Python + FastAPI** REST backend\n"
            "- **Unstructured data** — File parsing, preprocessing, "
            "three chunking strategies\n"
            "- **Embedding models** — multilingual semantic vector "
            "representations\n"
            "- **Vector database + hybrid search** — Qdrant (semantic) "
            "+ BM25 (lexical)\n"
            "- **Reranking** — cross-encoder for precise relevance\n"
            "- **RAG architecture** — LangChain + LangGraph-orchestrated pipeline\n"
            "- **LLM integration** — provider-agnostic, prompt-driven "
            "answer generation\n"
            "- **Evaluation** — RAGAS-based quality measurement\n"
            "- **Agent Flows** — LangGraph Plan-and-Execute with tool calling and conversation memory\n"
            "- [**Source documents**](https://github.com/Xiiqiing/Dokumentassistent/tree/main/docs)"
            " — the documents indexed into the knowledge base"
        ),
        "chunking_label": "Chunking strategy",
        "chunking_help": "Choose how documents are split into text chunks.",
        "topk_label": "Number of sources (top_k)",
        "topk_help": "Number of document fragments retrieved from the search index.",
        "title": "Document Intelligence System",
        "title_badge": "",
        "subtitle": (
            "A document intelligence system built on a RAG architecture, covering file ingestion, semantic chunking, "
            "hybrid retrieval with reranking, "
            "and LLM-generated answers with source citations. The LLM layer is provider-agnostic. "
            "Two modes: a LangGraph Plan-and-Execute agent (default) with conversation memory for complex multi-step queries, "
            "and a predefined pipeline for lightweight models. "
            "Retrieval quality is evaluated with RAGAS."
        ),
        "search_label": "Ask a question ...",
        "search_placeholder": "E.g.: What are the rules for processing personal data?",
        "search_button": "Search",
        "example_button": "Random question",
        "spinner": "Searching documents ...",
        "status_label": "Thinking",
        "status_done": "Done",
        "status_error": "Something went wrong",
        "confidence_label": "Confidence",
        "intent_label": "Intent",
        "strategy_label": "Strategy",
        "no_answer": "No answer received.",
        "sources_label": "Sources",
        "page_label": "page",
        "no_sources": "No sources found for this query.",
        "empty_warning": "Please enter a question.",
        "err_connection": (
            "Could not connect to the API server. "
            "Make sure the backend is running at http://localhost:8000."
        ),
        "err_api": "API error",
        "err_rate_limit": "Too many simultaneous requests, or API quota temporarily exhausted. Please wait a moment and try again.",
        "err_timeout": "The request took too long. Please try again.",
        "unknown": "unknown",
        "model_heading": "Current model",
        "model_llm": "LLM",
        "model_embedding": "Embedding",
        "model_unavailable": "Could not fetch model info.",
        "pipeline_heading": "Pipeline Details",
        "pipeline_translation": "Query Translation",
        "pipeline_original": "Original query",
        "pipeline_translated": "Translated to Danish",
        "pipeline_lang": "Detected language",
        "pipeline_no_translation": "No need for translation",
        "pipeline_bm25": "BM25 Results (lexical search)",
        "pipeline_dense": "Vector Search (semantic)",
        "pipeline_fused": "RRF Fused Ranking",
        "pipeline_reranked": "Reranked (final ranking)",
        "pipeline_doc": "Document",
        "pipeline_score": "Score",
        "pipeline_rank": "#",
        "pipeline_no_results": "No results",
        "pipeline_score_change": "Score change",
        "pipeline_plan_steps": "Execution Plan",
        "pipeline_tool_calls": "Tool Calls",
        "synthesize_status": "Synthesizing final answer ...",
        "example_note": "",
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dokumentintelligens-system",
    page_icon="📄",
    layout="centered",
)

st.markdown('<meta name="robots" content="noindex, nofollow">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Analytics — Umami Cloud
# ---------------------------------------------------------------------------
st.html(
    '<script async src="https://cloud.umami.is/script.js"'
    ' data-website-id="cf6c908e-1236-4406-8c02-88aa7c9a0db2"></script>',
)

# ---------------------------------------------------------------------------
# Custom CSS  --  Clean sans-serif design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #333333;
        background-color: #FFFFFF;
    }

    /* Hide default Streamlit branding but keep the sidebar toggle */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ---------- Accent line ---------- */
    .accent-line {
        width: 100%;
        height: 4px;
        background-color: #901A1E;
        margin-bottom: 1.5rem;
    }

    /* ---------- Title ---------- */
    .app-title {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #901A1E;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.02em;
        white-space: nowrap;
    }
    @media (max-width: 640px) {
        .app-title {
            font-size: clamp(1.3rem, 6vw, 2.2rem);
            white-space: nowrap;
        }
    }
    .app-subtitle {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 1.05rem;
        color: #666666;
        margin: 0 0 2rem 0;
        line-height: 1.6;
    }
    @media (max-width: 640px) {
        .app-subtitle {
            font-size: 0.82rem;
            line-height: 1.5;
        }
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E0E0E0;
    }
    /* Sidebar collapse button: always visible & KU red */
    button[data-testid="stBaseButton-headerNoPadding"] {
        opacity: 1 !important;
        visibility: visible !important;
        color: #901A1E !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        stroke: #901A1E !important;
        color: #901A1E !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]:hover {
        color: #6B1315 !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]:hover svg {
        stroke: #6B1315 !important;
        color: #6B1315 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    /* ---------- Shrink main area top gap to align title with sidebar heading ---------- */
    .block-container {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] .sidebar-heading {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #901A1E;
        margin-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        font-size: 0.92rem;
        color: #555555;
        line-height: 1.55;
    }
    section[data-testid="stSidebar"] ul {
        padding-left: 1.2rem;
        margin: 0.4rem 0 0 0;
        list-style-position: outside;
    }
    section[data-testid="stSidebar"] li {
        padding-left: 0.2rem;
        margin-bottom: 0.35rem;
    }

    /* ---------- Source card ---------- */
    .source-card {
        border: 1px solid #CCCCCC;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        background-color: #FAFAFA;
    }
    .source-card-title {
        font-weight: 600;
        color: #333333;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .source-card-text {
        font-size: 0.88rem;
        color: #555555;
        line-height: 1.55;
    }
    .source-card-meta {
        font-size: 0.8rem;
        color: #888888;
        margin-top: 0.4rem;
    }

    /* ---------- Result metadata ---------- */
    .result-meta {
        font-size: 0.88rem;
        color: #666666;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #E0E0E0;
    }

    /* ---------- Answer area ---------- */
    .answer-block {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #333333;
        margin-bottom: 1.5rem;
    }

    /* ---------- Search form container — equal padding on all sides ---------- */
    [data-testid="stForm"] {
        padding: 1.5rem !important;
    }

    /* ---------- Inputs ---------- */
    .stTextInput {
        margin-bottom: -0.5rem !important;
    }
    .stTextInput > div > div > input {
        border-radius: 0 !important;
        border: 1px solid #999999 !important;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #901A1E !important;
        box-shadow: none !important;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        border-radius: 0 !important;
        background-color: #901A1E !important;
        color: #FFFFFF !important;
        border: none !important;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 2rem !important;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        background-color: #7A1619 !important;
    }
    .stButton > button:active {
        background-color: #611114 !important;
    }

    /* ---------- Slider ---------- */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #901A1E !important;
    }

    /* ---------- Selectbox ---------- */
    .stSelectbox > div > div {
        border-radius: 0 !important;
    }

    /* ---------- Language toggle (the only st.radio on the page) ---------- */
    /* Collapse vertical space around the toggle row */
    [data-testid="stRadio"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] label {
        min-height: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Reduce gap between language row and accent line */
    [data-testid="stHorizontalBlock"]:first-child {
        margin-bottom: -0.8rem !important;
    }
    /* Hide the "Language" label */
    [data-testid="stRadio"] > label {
        display: none !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.15rem !important;
        justify-content: flex-end;
        align-items: center;
    }
    /* Hide the radio dot circle */
    [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    /* Base style for both options */
    [data-testid="stRadio"] [role="radiogroup"] label {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] label,
    [data-testid="stRadio"] [role="radiogroup"] label * {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #999999 !important;
        cursor: pointer;
        line-height: 1.2 !important;
    }
    /* Active / checked option → KU red */
    [data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"],
    [data-testid="stRadio"] [role="radiogroup"] label[data-checked="true"] *,
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
        color: #901A1E !important;
    }
    /* Separator between the two options */
    [data-testid="stRadio"] [role="radiogroup"] label:first-child::after {
        content: "|";
        color: #CCCCCC;
        font-weight: 400;
        margin-left: 0.5rem;
        font-size: 0.92rem;
    }

    /* ---------- Animated thinking dots on st.status label ---------- */
    @keyframes thinking-dots {
        from { width: 0; }
        to   { width: 1.2em; }
    }
    details[data-testid="stStatus"] > summary div[data-testid="stMarkdownContainer"] p::after {
        content: "...";
        display: inline-block;
        width: 0;
        overflow: hidden;
        vertical-align: bottom;
        white-space: nowrap;
        margin-left: 0.15em;
        animation: thinking-dots 1.2s steps(4, end) infinite;
    }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        font-size: 1rem !important;
        color: #333333 !important;
    }

    /* ---------- Footer ---------- */
    /* Clear transforms on ALL ancestors so position:fixed works */
    *:has(> .app-footer),
    *:has(.app-footer) {
        transform: none !important;
    }
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000001;
        background-color: #FFFFFF;
        text-align: center;
        border-top: 1px solid #E0E0E0;
        padding: 0.55rem 1.5rem;
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 0.82rem;
        color: #888888;
    }
    .app-footer a {
        color: #555555;
        text-decoration: none;
    }
    .app-footer a:hover {
        color: #901A1E;
    }
    .app-footer svg {
        vertical-align: middle;
        margin-right: 0.25rem;
    }
    .app-footer .footer-sep {
        margin: 0 0.6rem;
        color: #CCCCCC;
    }
    /* Push main content above the fixed footer */
    .block-container {
        padding-bottom: 4rem !important;
    }
    @media (max-width: 640px) {
        .app-footer {
            padding: 0.45rem 1rem;
            font-size: 0.78rem;
        }
        .block-container {
            padding-bottom: 5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Language selector  --  right-aligned toggle styled in KU red
# ---------------------------------------------------------------------------
_col_spacer, _col_lang = st.columns([5, 1.5])
with _col_lang:
    lang = st.radio(
        "Language",
        options=["da", "en"],
        format_func=lambda c: "Dansk" if c == "da" else "English",
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

t = TEXTS[lang]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-heading">{t["sidebar_heading"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(t["sidebar_body"])

    st.markdown("---")

    strategy = st.selectbox(
        t["chunking_label"],
        options=["fixed_size", "recursive", "semantic"],
        index=2,
        help=t["chunking_help"],
    )

    top_k = st.slider(
        t["topk_label"],
        min_value=1,
        max_value=20,
        value=5,
        help=t["topk_help"],
    )

    st.markdown("---")

    try:
        _health = requests.get(f"{API_BASE}/health", timeout=5).json()
        _llm = _health.get("llm_model", "")
        _llm_prov = _health.get("llm_provider", "")
        _emb = _health.get("embedding_model", "")
        _emb_prov = _health.get("embedding_provider", "")
        st.markdown(
            f'<div class="sidebar-heading">{t["model_heading"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**{t["model_llm"]}:** {_llm} ({_llm_prov})  \n'
            f'**{t["model_embedding"]}:** {_emb} ({_emb_prov})'
        )
    except Exception:
        st.caption(t["model_unavailable"])

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Accent line
st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

# Title block
st.markdown(
    f'<div class="app-title">{t["title"]}</div>',
    unsafe_allow_html=True,
)
# Subtitle placeholder — filled after we know whether search was clicked
_subtitle_slot = st.empty()

# ---------------------------------------------------------------------------
# Result rendering (extracted so it can be reused for cached results)
# ---------------------------------------------------------------------------
def _render_results(data: dict, t: dict, strategy: str, top_k: int) -> None:
    """Render query results: metadata bar, answer, sources, pipeline details."""
    confidence = data.get("confidence", 0.0)
    intent = data.get("intent", t["unknown"])
    confidence_pct = f"{confidence * 100:.0f}%"

    st.markdown(
        f'<div class="result-meta">'
        f'{t["confidence_label"]}: <strong>{confidence_pct}</strong> &nbsp;&middot;&nbsp; '
        f'{t["intent_label"]}: <strong>{intent}</strong> &nbsp;&middot;&nbsp; '
        f'{t["strategy_label"]}: <strong>{strategy}</strong> &nbsp;&middot;&nbsp; '
        f"top_k: <strong>{top_k}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    answer = data.get("answer", t["no_answer"])
    st.markdown(answer)

    sources = data.get("sources", [])
    if sources:
        with st.expander(f'{t["sources_label"]} ({len(sources)})', expanded=False):
            for src in sources:
                doc_name = src.get("document_id", src.get("chunk_id", t["unknown"]))
                text = src.get("text", "")
                score = src.get("score", 0.0)
                retrieval_source = src.get("source", "")
                metadata = src.get("metadata", {})
                page = metadata.get("page_number", "") if isinstance(metadata, dict) else ""

                page_info = f' &middot; {t["page_label"]} {page}' if page else ""
                score_display = f"{score:.3f}"

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-card-title">{html.escape(doc_name)}{page_info}</div>'
                    f'<div class="source-card-text">{html.escape(text[:500])}</div>'
                    f'<div class="source-card-meta">'
                    f"Score: {score_display} &nbsp;&middot;&nbsp; {html.escape(retrieval_source)}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info(t["no_sources"])

    pd_details = data.get("pipeline_details", {})
    if pd_details:
        with st.expander(t["pipeline_heading"], expanded=False):
            plan_steps = pd_details.get("plan_steps", [])
            if plan_steps:
                st.markdown(f'**{t["pipeline_plan_steps"]}**')
                for i, step_item in enumerate(plan_steps, 1):
                    st.markdown(f"{i}. {step_item}")
                st.markdown("---")

            tool_calls = pd_details.get("tool_calls", [])
            if tool_calls:
                st.markdown(f'**{t["pipeline_tool_calls"]}**')
                for tc in tool_calls:
                    st.markdown(f"- `{tc}`")
                st.markdown("---")

            if pd_details.get("translated"):
                st.markdown(f'**{t["pipeline_translation"]}**')
                st.markdown(
                    f'- {t["pipeline_lang"]}: **{pd_details.get("detected_language", "")}**\n'
                    f'- {t["pipeline_original"]}: {pd_details.get("original_query", "")}\n'
                    f'- {t["pipeline_translated"]}: {pd_details.get("retrieval_query", "")}'
                )
                st.markdown("---")

            def _truncate_doc(name: str, max_len: int = 30) -> str:
                return name if len(name) <= max_len else name[:max_len - 1] + "\u2026"

            def _render_result_table(results: list[dict], label: str) -> None:
                st.markdown(f"**{label}**")
                if not results:
                    st.caption(t["pipeline_no_results"])
                    return
                header = f'| {t["pipeline_rank"]} | {t["pipeline_doc"]} | {t["pipeline_score"]} |\n|---|---|---|'
                rows = "\n".join(
                    f'| {i + 1} | {_truncate_doc(r.get("document_id", ""))} | {r.get("score", 0):.4f} |'
                    for i, r in enumerate(results)
                )
                st.markdown(f"{header}\n{rows}")

            _has_retrieval = bool(
                pd_details.get("dense_results") or pd_details.get("sparse_results") or pd_details.get("fused_results")
            )

            if _has_retrieval:
                _render_result_table(pd_details.get("sparse_results", []), t["pipeline_bm25"])
                st.markdown("---")
                _render_result_table(pd_details.get("dense_results", []), t["pipeline_dense"])
                st.markdown("---")
                _render_result_table(pd_details.get("fused_results", []), t["pipeline_fused"])
                st.markdown("---")

            reranked = pd_details.get("reranked_results", [])
            st.markdown(f'**{t["pipeline_reranked"]}**')
            if reranked:
                if _has_retrieval:
                    fused_scores: dict[str, float] = {
                        r.get("chunk_id", ""): r.get("score", 0.0)
                        for r in pd_details.get("fused_results", [])
                    }
                    header = (
                        f'| {t["pipeline_rank"]} | {t["pipeline_doc"]} | '
                        f'{t["pipeline_score"]} | {t["pipeline_score_change"]} |\n'
                        f"|---|---|---|---|"
                    )
                    rows_list = []
                    for i, r in enumerate(reranked):
                        cid = r.get("chunk_id", "")
                        new_score = r.get("score", 0.0)
                        old_score = fused_scores.get(cid)
                        if old_score is not None:
                            change = f"RRF {old_score:.4f} -> {new_score:.4f}"
                        else:
                            change = "-"
                        rows_list.append(
                            f'| {i + 1} | {_truncate_doc(r.get("document_id", ""))} | {new_score:.4f} | {change} |'
                        )
                    st.markdown(f"{header}\n" + "\n".join(rows_list))
                else:
                    header = f'| {t["pipeline_rank"]} | {t["pipeline_doc"]} | {t["pipeline_score"]} |\n|---|---|---|'
                    rows = "\n".join(
                        f'| {i + 1} | {_truncate_doc(r.get("document_id", ""))} | {r.get("score", 0):.4f} |'
                        for i, r in enumerate(reranked)
                    )
                    st.markdown(f"{header}\n{rows}")
            else:
                st.caption(t["pipeline_no_results"])


# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------
def _pick_example() -> None:
    """Select a random example question and store it in session state."""
    st.session_state.query_input = random.choice(EXAMPLE_QUESTIONS)

with st.form(key="search_form", clear_on_submit=False):
    question = st.text_input(
        t["search_label"],
        key="query_input",
        placeholder=t["search_placeholder"],
    )
    col_search, col_example = st.columns([1, 1])
    with col_search:
        search_clicked = st.form_submit_button(t["search_button"], use_container_width=True)
    with col_example:
        st.form_submit_button(t["example_button"], on_click=_pick_example, use_container_width=True)
        if t["example_note"]:
            st.markdown(
                f'<div style="text-align:right;font-size:0.85em;color:gray;">{t["example_note"]}</div>',
                unsafe_allow_html=True,
            )

# Show subtitle only when no search is active
if not search_clicked and not st.session_state.get("has_searched"):
    _subtitle_slot.markdown(
        f'<div class="app-subtitle">{t["subtitle"]}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------
if search_clicked and question.strip():
    st.session_state["has_searched"] = True
    data: dict = {}
    _sse_error: dict | None = None

    with st.status(t["status_label"], expanded=True) as _status:
        try:
            with requests.post(
                f"{API_BASE}/query/stream",
                json={
                    "question": question.strip(),
                    "top_k": top_k,
                    "strategy": strategy,
                    "session_id": st.session_state["session_id"],
                },
                stream=True,
                timeout=180,
            ) as _resp:
                _resp.raise_for_status()

                for _raw in _resp.iter_lines():
                    if not _raw:
                        continue
                    _line = _raw.decode("utf-8") if isinstance(_raw, bytes) else _raw
                    if not _line.startswith("data: "):
                        continue

                    try:
                        _event = json.loads(_line[6:])
                    except json.JSONDecodeError:
                        continue
                    _step = _event.get("step", "")

                    if _step == "detect":
                        _intent_val = _event.get("intent", "")
                        _lang_val = _event.get("language", "")
                        if lang == "da":
                            st.write(f"Intent: **{_intent_val}** · Sprog: **{_lang_val}**")
                        else:
                            st.write(f"Intent: **{_intent_val}** · Language: **{_lang_val}**")

                    elif _step == "translate":
                        if _event.get("translated"):
                            _rq = _event.get("retrieval_query", "")
                            st.write(
                                (f"Oversat til dansk: _{_rq}_")
                                if lang == "da"
                                else (f"Translated to Danish: _{_rq}_")
                            )
                        else:
                            st.write(
                                "Ingen oversættelse nødvendig for forespørgslen"
                                if lang == "da"
                                else "No translation needed for the query"
                            )

                    elif _step == "retrieve":
                        _dc = _event.get("dense_count", 0)
                        _sc = _event.get("sparse_count", 0)
                        st.write(
                            (f"Fandt **{_dc}** semantiske + **{_sc}** leksikalske kandidater")
                            if lang == "da"
                            else (f"Found **{_dc}** semantic + **{_sc}** lexical candidates")
                        )

                    elif _step == "rerank":
                        _rc = _event.get("reranked_count", 0)
                        _cf = _event.get("confidence", 0.0)
                        st.write(
                            (f"Reranket til **{_rc}** resultater · konfidensgrad **{_cf:.0%}**")
                            if lang == "da"
                            else (f"Reranked to **{_rc}** results · confidence **{_cf:.0%}**")
                        )

                    elif _step == "plan":
                        _steps = _event.get("steps", [])
                        st.write(
                            (f"Plan oprettet med **{len(_steps)}** trin")
                            if lang == "da"
                            else (f"Plan created with **{len(_steps)}** steps")
                        )
                        for _ps in _steps:
                            st.write(f"  - {_ps}")

                    elif _step == "execute_step":
                        _si = _event.get("step_index", 0)
                        _sd = _event.get("step_desc", "")
                        st.write(
                            (f"Trin {_si} udført: _{_sd}_")
                            if lang == "da"
                            else (f"Step {_si} executed: _{_sd}_")
                        )

                    elif _step == "synthesize":
                        st.write(t["synthesize_status"])

                    elif _step == "tool_call":
                        _tool_name = _event.get("tool", "")
                        _tool_query = _event.get("query", "")
                        if _tool_query:
                            st.write(
                                (f"Værktøj **{_tool_name}** kaldt: _{_tool_query}_")
                                if lang == "da"
                                else (f"Tool **{_tool_name}** called: _{_tool_query}_")
                            )
                        else:
                            st.write(
                                (f"Værktøj **{_tool_name}** kaldt")
                                if lang == "da"
                                else (f"Tool **{_tool_name}** called")
                            )

                    elif _step == "tool_result":
                        _rc = _event.get("result_count", 0)
                        _tool_name = _event.get("tool", "")
                        if _tool_name == "list_documents":
                            # list_documents returns doc list in its text,
                            # parse count from the tool output or show generic
                            st.write(
                                "Dokumentliste hentet"
                                if lang == "da"
                                else "Document list retrieved"
                            )
                        elif _tool_name == "fetch_document":
                            st.write(
                                (f"Hentet dokument (**{_rc}** afsnit)")
                                if lang == "da"
                                else (f"Fetched document (**{_rc}** chunks)")
                            )
                        else:
                            st.write(
                                (f"Fandt **{_rc}** relevante passager")
                                if lang == "da"
                                else (f"Found **{_rc}** relevant passages")
                            )

                    elif _step == "broaden_query":
                        _retry = _event.get("retry_count", 1)
                        _rq = _event.get("retrieval_query", "")
                        st.write(
                            (f"Lav konfidensgrad – forsøg {_retry} med udvidet søgning: _{_rq}_")
                            if lang == "da"
                            else (f"Low confidence – retry {_retry} with broadened query: _{_rq}_")
                        )

                    elif _step == "generate":
                        st.write(
                            "Svar genereret"
                            if lang == "da"
                            else "Answer generated"
                        )

                    elif _step == "rate_limit":
                        _rl_msg = _event.get("message", "")
                        st.warning(
                            f"⏳ {_rl_msg} — vent venligst ..."
                            if lang == "da"
                            else f"⏳ {_rl_msg} — please wait ..."
                        )

                    elif _step == "done":
                        data = _event.get("result", {})
                        _status.update(label=t["status_done"], state="complete", expanded=False)

                    elif _step == "error":
                        _sse_error = _event
                        _status.update(label=t["status_error"], state="error", expanded=True)
                        break

        except requests.ConnectionError:
            _status.update(label=t["status_error"], state="error", expanded=True)
            st.error(t["err_connection"])
            st.stop()
        except requests.HTTPError as _exc:
            _status.update(label=t["status_error"], state="error", expanded=True)
            if _exc.response.status_code == 429:
                st.warning(t["err_rate_limit"])
            else:
                st.error(f'{t["err_api"]}: {_exc.response.status_code} -- {_exc.response.text}')
            st.stop()
        except requests.Timeout:
            _status.update(label=t["status_error"], state="error", expanded=True)
            st.error(t["err_timeout"])
            st.stop()

    if _sse_error is not None:
        if _sse_error.get("code") == 429:
            st.warning(t["err_rate_limit"])
        else:
            st.error(f'{t["err_api"]}: {_sse_error.get("message", "")}')
        st.stop()

    # Cache result in session_state so it survives iOS tombstone / reconnect
    st.session_state["last_result"] = data
    st.session_state["last_question"] = question.strip()
    st.session_state["last_strategy"] = strategy
    st.session_state["last_top_k"] = top_k

    _render_results(data, t, strategy, top_k)

elif search_clicked:
    st.warning(t["empty_warning"])

elif "last_result" in st.session_state:
    # Restore cached results after iOS tombstone / reconnect
    _render_results(
        st.session_state["last_result"],
        t,
        st.session_state.get("last_strategy", strategy),
        st.session_state.get("last_top_k", top_k),
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <a href="https://github.com/Xiiqiing/Dokumentassistent" target="_blank" rel="noopener noreferrer">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>
            Xiiqiing/Dokumentassistent
        </a>
        <span class="footer-sep">|</span>
        <span>&copy; 2026 Xiqing</span>
    </div>
    """,
    unsafe_allow_html=True,
)
