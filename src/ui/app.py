"""Streamlit frontend for Dokumentassistent.

Calls the FastAPI backend at http://localhost:8000.
Single-page document search interface inspired by Danish university design.
"""

import os
from typing import Dict

import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Internationalisation — all UI strings live here
# ---------------------------------------------------------------------------
TEXTS: Dict[str, Dict[str, str]] = {
    "da": {
        "page_title": "Dokumentassistent",
        "lang_label": "Sprog",
        "sidebar_heading": "Om systemet",
        "sidebar_body": (
            "End-to-end RAG-prototype der goer dansksproget "
            "dokumenthaandtering selvbetjent.\n\n"
            "- **Python + FastAPI** REST-backend\n"
            "- **Ustruktureret data** — PDF-parsing, preprocessing, "
            "tre chunking-strategier\n"
            "- **Embedding-modeller** — flersproget semantisk "
            "vektorrepraesentation\n"
            "- **Vektordatabase + hybrid soegning** — Qdrant (semantisk) "
            "+ BM25 (leksikalsk)\n"
            "- **Reranking** — cross-encoder for praecis relevans\n"
            "- **RAG-arkitektur** — LangChain-orkestreret pipeline\n"
            "- **LLM-integration** — provider-agnostisk, prompt-styret "
            "svargenerering\n"
            "- **Evaluering** — RAGAS-baseret kvalitetsmaaling\n"
            "- **Agent-routing** — intent-klassifikation og "
            "forespørgselsdirigering"
        ),
        "chunking_label": "Chunking-strategi",
        "chunking_help": "Vaelg hvordan dokumenterne opdeles i tekststykker.",
        "topk_label": "Antal kilder (top_k)",
        "topk_help": "Antal dokumentfragmenter der hentes fra sogeindekset.",
        "title": "Dokumentassistent",
        "subtitle": (
            "Stil et sporgsmål, og systemet finder relevante afsnit "
            "i eksempelfiler (såsom politiske dokumenter fra KU)."
        ),
        "search_label": "Stil et sporgsmål om ... ",
        "search_placeholder": "F.eks.: Hvad er reglerne for behandling af personoplysninger?",
        "search_button": "Sog",
        "spinner": "Soger i dokumenterne ...",
        "confidence_label": "Konfidensgrad",
        "intent_label": "Intent",
        "strategy_label": "Strategi",
        "no_answer": "Intet svar modtaget.",
        "sources_label": "Kilder",
        "page_label": "side",
        "no_sources": "Ingen kilder fundet for denne foresporgsel.",
        "empty_warning": "Indtast venligst et sporgsmål.",
        "err_connection": (
            "Kunne ikke oprette forbindelse til API-serveren. "
            "Kontroller at backend korer paa http://localhost:8000."
        ),
        "err_api": "API-fejl",
        "err_timeout": "Forespoorgslen tog for lang tid. Prøv igen.",
        "unknown": "ukendt",
        "model_heading": "Aktuel model",
        "model_llm": "LLM",
        "model_embedding": "Embedding",
        "model_unavailable": "Kunne ikke hente modelinfo.",
        "pipeline_heading": "Pipeline-detaljer",
        "pipeline_translation": "Oversaettelse",
        "pipeline_original": "Original foresporgsel",
        "pipeline_translated": "Oversat til dansk",
        "pipeline_lang": "Sprog registreret",
        "pipeline_no_translation": "Ingen oversaettelse (foresporgsel allerede paa dansk)",
        "pipeline_bm25": "BM25-resultater (leksikalsk soegning)",
        "pipeline_dense": "Vektorsoegning (semantisk)",
        "pipeline_fused": "RRF-fusioneret raekkefoelge",
        "pipeline_reranked": "Reranking (endelig raekkefoelge)",
        "pipeline_doc": "Dokument",
        "pipeline_score": "Score",
        "pipeline_rank": "#",
        "pipeline_no_results": "Ingen resultater",
        "pipeline_score_change": "Score-aendring",
        "nav_home": "Forside",
        "nav_docs": "Dokumenter",
        "nav_search": "Soegning",
        "nav_about": "Om systemet",
        "breadcrumb_home": "Forside",
        "breadcrumb_current": "Dokumentassistent",
        "footer_contact": "Kontakt",
        "footer_contact_text": "IT-support for administrativt personale",
        "footer_services": "Services",
        "footer_service_api": "API-dokumentation",
        "footer_service_status": "Systemstatus",
        "footer_service_guide": "Brugervejledning",
        "footer_about": "Om",
        "footer_about_privacy": "Privatlivspolitik",
        "footer_about_data": "Databehandling",
        "footer_about_access": "Tilgaengelighed",
        "footer_copyright": "Dokumentassistent — RAG-prototype",
    },
    "en": {
        "page_title": "Document Assistant",
        "lang_label": "Language",
        "sidebar_heading": "About the system",
        "sidebar_body": (
            "End-to-end RAG prototype that makes Danish-language "
            "document Q&A self-service.\n\n"
            "- **Python + FastAPI** REST backend\n"
            "- **Unstructured data** — PDF parsing, preprocessing, "
            "three chunking strategies\n"
            "- **Embedding models** — multilingual semantic vector "
            "representations\n"
            "- **Vector database + hybrid search** — Qdrant (semantic) "
            "+ BM25 (lexical)\n"
            "- **Reranking** — cross-encoder for precise relevance\n"
            "- **RAG architecture** — LangChain-orchestrated pipeline\n"
            "- **LLM integration** — provider-agnostic, prompt-driven "
            "answer generation\n"
            "- **Evaluation** — RAGAS-based quality measurement\n"
            "- **Agent routing** — intent classification and query "
            "dispatch"
        ),
        "chunking_label": "Chunking strategy",
        "chunking_help": "Choose how documents are split into text chunks.",
        "topk_label": "Number of sources (top_k)",
        "topk_help": "Number of document fragments retrieved from the search index.",
        "title": "Document Assistant",
        "subtitle": (
            "Ask a question, and the system will find relevant sections "
            "in example documents."
        ),
        "search_label": "Ask a question ...",
        "search_placeholder": "E.g.: What are the rules for processing personal data?",
        "search_button": "Search",
        "spinner": "Searching documents ...",
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
        "pipeline_no_translation": "No translation (query already in Danish)",
        "pipeline_bm25": "BM25 Results (lexical search)",
        "pipeline_dense": "Vector Search (semantic)",
        "pipeline_fused": "RRF Fused Ranking",
        "pipeline_reranked": "Reranked (final ranking)",
        "pipeline_doc": "Document",
        "pipeline_score": "Score",
        "pipeline_rank": "#",
        "pipeline_no_results": "No results",
        "pipeline_score_change": "Score change",
        "nav_home": "Home",
        "nav_docs": "Documents",
        "nav_search": "Search",
        "nav_about": "About",
        "breadcrumb_home": "Home",
        "breadcrumb_current": "Document Assistant",
        "footer_contact": "Contact",
        "footer_contact_text": "IT support for administrative staff",
        "footer_services": "Services",
        "footer_service_api": "API Documentation",
        "footer_service_status": "System Status",
        "footer_service_guide": "User Guide",
        "footer_about": "About",
        "footer_about_privacy": "Privacy Policy",
        "footer_about_data": "Data Processing",
        "footer_about_access": "Accessibility",
        "footer_copyright": "Document Assistant — RAG Prototype",
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dokumentassistent",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS  --  University-inspired visual identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
        color: #333333;
        background-color: #FFFFFF;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent; height: 0;}

    /* Force hide Streamlit header space */
    .stApp > header {display: none;}
    .block-container {
        padding-top: 0 !important;
        max-width: 1100px;
    }

    /* ---------- Top utility bar ---------- */
    .top-utility-bar {
        background-color: #2B2B2B;
        color: #CCCCCC;
        font-size: 0.78rem;
        padding: 6px 0;
        margin: -1rem -4rem 0 -4rem;
        width: 100vw;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
    }
    .top-utility-inner {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 2rem;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1.5rem;
    }
    .top-utility-bar a {
        color: #CCCCCC;
        text-decoration: none;
        transition: color 0.15s;
    }
    .top-utility-bar a:hover {
        color: #FFFFFF;
    }
    .utility-lang-switch {
        border-left: 1px solid #555;
        padding-left: 1.5rem;
    }

    /* ---------- Main navigation bar ---------- */
    .main-nav-bar {
        background-color: #FFFFFF;
        border-bottom: 3px solid #901A1E;
        margin: 0 -4rem;
        width: 100vw;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
    }
    .main-nav-inner {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0.9rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .nav-brand {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #901A1E;
        text-decoration: none;
        letter-spacing: -0.02em;
    }
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    .nav-links a {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 0.92rem;
        color: #333333;
        text-decoration: none;
        font-weight: 500;
        padding: 0.3rem 0;
        border-bottom: 2px solid transparent;
        transition: border-color 0.15s, color 0.15s;
    }
    .nav-links a:hover, .nav-links a.active {
        color: #901A1E;
        border-bottom-color: #901A1E;
    }

    /* ---------- Breadcrumbs ---------- */
    .breadcrumbs {
        font-size: 0.82rem;
        color: #888888;
        padding: 0.7rem 0 0.4rem 0;
        margin-bottom: 0.2rem;
    }
    .breadcrumbs a {
        color: #901A1E;
        text-decoration: none;
    }
    .breadcrumbs a:hover {
        text-decoration: underline;
    }
    .breadcrumbs .separator {
        color: #AAAAAA;
        margin: 0 0.4rem;
    }

    /* ---------- Page title area ---------- */
    .page-title-area {
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 1.2rem;
        margin-bottom: 1.8rem;
    }
    .page-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2B2B2B;
        margin: 0.4rem 0 0.5rem 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .page-subtitle {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 1rem;
        color: #666666;
        margin: 0;
        line-height: 1.6;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #F7F7F7;
        border-right: 1px solid #E0E0E0;
    }
    section[data-testid="stSidebar"] .sidebar-section-heading {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #2B2B2B;
        margin-bottom: 0.5rem;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #901A1E;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        font-size: 0.88rem;
        color: #555555;
        line-height: 1.55;
    }

    /* ---------- Source card ---------- */
    .source-card {
        border-left: 3px solid #901A1E;
        border-top: 1px solid #E0E0E0;
        border-right: 1px solid #E0E0E0;
        border-bottom: 1px solid #E0E0E0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        background-color: #FAFAFA;
        transition: box-shadow 0.15s;
    }
    .source-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .source-card-title {
        font-weight: 600;
        color: #2B2B2B;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .source-card-text {
        font-size: 0.85rem;
        color: #555555;
        line-height: 1.55;
    }
    .source-card-meta {
        font-size: 0.78rem;
        color: #888888;
        margin-top: 0.4rem;
    }

    /* ---------- Result metadata ---------- */
    .result-meta {
        font-size: 0.85rem;
        color: #666666;
        margin-bottom: 1.2rem;
        padding: 0.6rem 0.9rem;
        background-color: #F7F7F7;
        border-left: 3px solid #901A1E;
    }

    /* ---------- Answer area ---------- */
    .answer-block {
        font-size: 1.02rem;
        line-height: 1.7;
        color: #333333;
        margin-bottom: 1.5rem;
        padding: 1.2rem;
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
    }

    /* ---------- Inputs ---------- */
    .stTextInput > div > div > input {
        border-radius: 0 !important;
        border: 1px solid #BBBBBB !important;
        font-family: Arial, Helvetica, sans-serif !important;
        padding: 0.6rem 0.8rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #901A1E !important;
        box-shadow: 0 0 0 1px #901A1E !important;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        border-radius: 0 !important;
        background-color: #901A1E !important;
        color: #FFFFFF !important;
        border: none !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 2.2rem !important;
        letter-spacing: 0.02em;
        font-weight: 600 !important;
        text-transform: uppercase;
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

    /* ---------- Pipeline tables ---------- */
    [data-testid="stExpander"] table {
        table-layout: fixed;
        width: 100%;
        word-break: break-all;
        overflow-wrap: break-word;
    }
    [data-testid="stExpander"] table td,
    [data-testid="stExpander"] table th {
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 0;
        font-size: 0.82rem;
        padding: 0.3rem 0.5rem;
    }
    [data-testid="stExpander"] table td:first-child,
    [data-testid="stExpander"] table th:first-child {
        width: 2.5rem;
    }
    [data-testid="stExpander"] table td:last-child,
    [data-testid="stExpander"] table th:last-child {
        width: 5rem;
    }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 0.95rem !important;
        color: #333333 !important;
    }

    /* ---------- Footer ---------- */
    .site-footer {
        background-color: #2B2B2B;
        color: #CCCCCC;
        margin: 3rem -4rem 0 -4rem;
        width: 100vw;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
        padding: 2.5rem 0 1.5rem 0;
    }
    .footer-inner {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 2rem;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 2rem;
    }
    .footer-col h4 {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #901A1E;
        display: inline-block;
    }
    .footer-col p, .footer-col a {
        font-size: 0.82rem;
        color: #AAAAAA;
        line-height: 1.7;
        text-decoration: none;
        display: block;
    }
    .footer-col a:hover {
        color: #FFFFFF;
    }
    .footer-bottom {
        max-width: 1100px;
        margin: 1.5rem auto 0 auto;
        padding: 1rem 2rem 0 2rem;
        border-top: 1px solid #444444;
        font-size: 0.78rem;
        color: #888888;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Language state — use query param trick for the utility-bar lang switch
# ---------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "da"


def _set_lang(code: str) -> None:
    """Set the language in session state."""
    st.session_state.lang = code


# A hidden selectbox drives the actual state; the visual switch is in the bar
_col_hidden_lang = st.columns([1])[0]
with _col_hidden_lang:
    lang = st.selectbox(
        "lang_sel",
        options=["da", "en"],
        format_func=lambda c: "Dansk" if c == "da" else "English",
        index=0 if st.session_state.lang == "da" else 1,
        label_visibility="collapsed",
        key="lang_select",
    )
    st.session_state.lang = lang

t = TEXTS[lang]

# ---------------------------------------------------------------------------
# Top utility bar
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="top-utility-bar">
      <div class="top-utility-inner">
        <span>{t["footer_service_status"]}</span>
        <span>{t["footer_service_guide"]}</span>
        <span class="utility-lang-switch">{"Dansk" if lang == "da" else "English"}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main navigation bar
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-nav-bar">
      <div class="main-nav-inner">
        <span class="nav-brand">Dokumentassistent</span>
        <div class="nav-links">
          <a href="#" class="active">{t["nav_home"]}</a>
          <a href="#">{t["nav_docs"]}</a>
          <a href="#">{t["nav_search"]}</a>
          <a href="#">{t["nav_about"]}</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Breadcrumbs
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="breadcrumbs">
      <a href="#">{t["breadcrumb_home"]}</a>
      <span class="separator">&rsaquo;</span>
      <a href="#">{t["nav_search"]}</a>
      <span class="separator">&rsaquo;</span>
      <span>{t["breadcrumb_current"]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-section-heading">{t["sidebar_heading"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(t["sidebar_body"])

    st.markdown("---")

    strategy = st.selectbox(
        t["chunking_label"],
        options=["recursive", "semantic", "sliding"],
        index=0,
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
            f'<div class="sidebar-section-heading">{t["model_heading"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**{t["model_llm"]}:** {_llm} ({_llm_prov})  \n'
            f'**{t["model_embedding"]}:** {_emb} ({_emb_prov})'
        )
    except Exception:
        st.caption(t["model_unavailable"])

# ---------------------------------------------------------------------------
# Main content — page title area
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="page-title-area">
      <div class="page-title">{t["title"]}</div>
      <p class="page-subtitle">{t["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------
with st.form("search_form"):
    question = st.text_input(
        t["search_label"],
        placeholder=t["search_placeholder"],
    )
    search_clicked = st.form_submit_button(t["search_button"])

# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------
if search_clicked and question.strip():
    with st.spinner(t["spinner"]):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={
                    "question": question.strip(),
                    "top_k": top_k,
                    "strategy": strategy,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            st.error(t["err_connection"])
            st.stop()
        except requests.HTTPError as exc:
            st.error(f'{t["err_api"]}: {exc.response.status_code} -- {exc.response.text}')
            st.stop()
        except requests.Timeout:
            st.error(t["err_timeout"])
            st.stop()

    # -- Metadata bar --
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

    # -- Answer --
    answer = data.get("answer", t["no_answer"])
    st.markdown(f'<div class="answer-block">{answer}</div>', unsafe_allow_html=True)

    # -- Sources --
    sources = data.get("sources", [])
    if sources:
        with st.expander(f'{t["sources_label"]} ({len(sources)})', expanded=False):
            for src in sources:
                doc_name = src.get("document_id", src.get("chunk_id", t["unknown"]))
                text = src.get("text", "")
                score = src.get("score", 0.0)
                retrieval_source = src.get("source", "")
                metadata = src.get("metadata", {})
                page = metadata.get("page", "") if isinstance(metadata, dict) else ""

                page_info = f' &middot; {t["page_label"]} {page}' if page else ""
                score_display = f"{score:.3f}"

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-card-title">{doc_name}{page_info}</div>'
                    f'<div class="source-card-text">{text[:500]}</div>'
                    f'<div class="source-card-meta">'
                    f"Score: {score_display} &nbsp;&middot;&nbsp; {retrieval_source}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info(t["no_sources"])

    # -- Pipeline Details --
    pd = data.get("pipeline_details", {})
    if pd:
        with st.expander(t["pipeline_heading"], expanded=False):
            # 1) Query translation
            st.markdown(f'**{t["pipeline_translation"]}**')
            if pd.get("translated"):
                st.markdown(
                    f'- {t["pipeline_lang"]}: **{pd.get("detected_language", "")}**\n'
                    f'- {t["pipeline_original"]}: {pd.get("original_query", "")}\n'
                    f'- {t["pipeline_translated"]}: {pd.get("retrieval_query", "")}'
                )
            else:
                st.markdown(f'_{t["pipeline_no_translation"]}_')

            st.markdown("---")

            def _truncate_doc_id(doc_id: str, max_len: int = 40) -> str:
                """Truncate long document IDs for table display."""
                if len(doc_id) <= max_len:
                    return doc_id
                return doc_id[: max_len - 3] + "..."

            def _render_result_table(results: list[dict], label: str) -> None:
                """Render a ranked results table."""
                st.markdown(f"**{label}**")
                if not results:
                    st.caption(t["pipeline_no_results"])
                    return
                header = f'| {t["pipeline_rank"]} | {t["pipeline_doc"]} | {t["pipeline_score"]} |\n|---|---|---|'
                rows = "\n".join(
                    f'| {i + 1} | {_truncate_doc_id(r.get("document_id", ""))} | {r.get("score", 0):.4f} |'
                    for i, r in enumerate(results)
                )
                st.markdown(f"{header}\n{rows}")

            # 2) BM25 results
            _render_result_table(pd.get("sparse_results", []), t["pipeline_bm25"])

            st.markdown("---")

            # 3) Vector search results
            _render_result_table(pd.get("dense_results", []), t["pipeline_dense"])

            st.markdown("---")

            # 4) RRF fused ranking
            _render_result_table(pd.get("fused_results", []), t["pipeline_fused"])

            st.markdown("---")

            # 5) Reranked results with score change
            reranked = pd.get("reranked_results", [])
            st.markdown(f'**{t["pipeline_reranked"]}**')
            if reranked:
                # Build a map from chunk_id -> fused score for comparison
                fused_scores: dict[str, float] = {
                    r.get("chunk_id", ""): r.get("score", 0.0)
                    for r in pd.get("fused_results", [])
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
                        f'| {i + 1} | {_truncate_doc_id(r.get("document_id", ""))} | {new_score:.4f} | {change} |'
                    )
                st.markdown(f"{header}\n" + "\n".join(rows_list))
            else:
                st.caption(t["pipeline_no_results"])

elif search_clicked:
    st.warning(t["empty_warning"])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="site-footer">
      <div class="footer-inner">
        <div class="footer-col">
          <h4>Dokumentassistent</h4>
          <p>{t["footer_contact_text"]}</p>
          <p style="margin-top:0.5rem; color:#888;">RAG-pipeline &middot; Hybrid Search &middot; LLM</p>
        </div>
        <div class="footer-col">
          <h4>{t["footer_services"]}</h4>
          <a href="#">{t["footer_service_api"]}</a>
          <a href="#">{t["footer_service_status"]}</a>
          <a href="#">{t["footer_service_guide"]}</a>
        </div>
        <div class="footer-col">
          <h4>{t["footer_about"]}</h4>
          <a href="#">{t["footer_about_privacy"]}</a>
          <a href="#">{t["footer_about_data"]}</a>
          <a href="#">{t["footer_about_access"]}</a>
        </div>
        <div class="footer-col">
          <h4>{t["footer_contact"]}</h4>
          <p>support@example.dk</p>
          <p>+45 00 00 00 00</p>
        </div>
      </div>
      <div class="footer-bottom">
        {t["footer_copyright"]}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
