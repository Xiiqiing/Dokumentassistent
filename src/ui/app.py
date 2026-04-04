"""Streamlit frontend for Dokumentassistent.

Calls the FastAPI backend at http://localhost:8000.
Single-page document search interface with clean sans-serif design.
"""

import os

import streamlit as st
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Internationalisation — all UI strings live here
# ---------------------------------------------------------------------------
TEXTS: dict[str, dict[str, str]] = {
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
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dokumentassistent",
    page_icon=None,
    layout="centered",
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
    }
    .app-subtitle {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 1.05rem;
        color: #666666;
        margin: 0 0 2rem 0;
        line-height: 1.6;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #E0E0E0;
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

    /* ---------- Inputs ---------- */
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

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        font-size: 1rem !important;
        color: #333333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Language selector  --  top-right corner via columns
# ---------------------------------------------------------------------------
_col_spacer, _col_lang = st.columns([5, 1])
with _col_lang:
    lang = st.selectbox(
        "🌐",
        options=["da", "en"],
        format_func=lambda c: "Dansk" if c == "da" else "English",
        index=0,
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
st.markdown(f'<div class="app-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="app-subtitle">{t["subtitle"]}</div>',
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
            # 1) Query translation (only show if translation actually happened)
            if pd.get("translated"):
                st.markdown(f'**{t["pipeline_translation"]}**')
                st.markdown(
                    f'- {t["pipeline_lang"]}: **{pd.get("detected_language", "")}**\n'
                    f'- {t["pipeline_original"]}: {pd.get("original_query", "")}\n'
                    f'- {t["pipeline_translated"]}: {pd.get("retrieval_query", "")}'
                )
                st.markdown("---")

            def _render_result_table(results: list[dict], label: str) -> None:
                """Render a ranked results table."""
                st.markdown(f"**{label}**")
                if not results:
                    st.caption(t["pipeline_no_results"])
                    return
                header = f'| {t["pipeline_rank"]} | {t["pipeline_doc"]} | {t["pipeline_score"]} |\n|---|---|---|'
                rows = "\n".join(
                    f'| {i + 1} | {r.get("document_id", "")} | {r.get("score", 0):.4f} |'
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
                        f'| {i + 1} | {r.get("document_id", "")} | {new_score:.4f} | {change} |'
                    )
                st.markdown(f"{header}\n" + "\n".join(rows_list))
            else:
                st.caption(t["pipeline_no_results"])

elif search_clicked:
    st.warning(t["empty_warning"])
