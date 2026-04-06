"""Streamlit frontend for Dokumentassistent.

Calls the FastAPI backend at http://localhost:8000.
Single-page document search interface with clean sans-serif design.
"""

import json
import os
import random

import streamlit as st
import streamlit.components.v1 as components
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

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
        "page_title": "Dokumentassistent",
        "lang_label": "Sprog",
        "sidebar_heading": "Om systemet",
        "sidebar_body": (
            "- **Python + FastAPI** REST-backend\n"
            "- **Ustruktureret data** — PDF-parsing, preprocessing, "
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
            "- **Agent Flows** — ReAct-loop med værktøjskald."
        ),
        "chunking_label": "Chunking-strategi",
        "chunking_help": "Vælg hvordan dokumenterne opdeles i tekststykker.",
        "topk_label": "Antal kilder (top_k)",
        "topk_help": "Antal dokumentfragmenter der hentes fra søgeindekset.",
        "title": "Dokumentassistent",
        "title_badge": "Demo",
        "subtitle": (
            "Et dokumentintelligens-system der dækker PDF-indlæsning, semantisk chunking, "
            "hybrid søgning med reranking "
            "og LLM-genererede svar med kildehenvisninger. LLM-laget er provider-agnostisk. "
            "To tilstande: en fast pipeline til lette modeller og en LangGraph ReAct-agent "
            "til forespørgsler der kræver flere søgetrin. Søgekvaliteten evalueres med RAGAS."
        ),
        "search_label": "Stil et spørgsmål om ... ",
        "search_placeholder": "F.eks.: Hvad er reglerne for behandling af personoplysninger?",
        "search_button": "Søg",
        "example_button": "Tilfældigt eksempel",
        "spinner": "Søger i dokumenterne ...",
        "status_label": "Behandler forespørgsel ...",
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
        "err_rate_limit": "API-kvoten er midlertidigt opbrugt. Vent venligst et øjeblik, og prøv igen.",
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
    },
    "en": {
        "page_title": "Document Assistant",
        "lang_label": "Language",
        "sidebar_heading": "About the system",
        "sidebar_body": (
            "- **Python + FastAPI** REST backend\n"
            "- **Unstructured data** — PDF parsing, preprocessing, "
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
            "- **Agent Flows** — ReAct loop with tool calling"
        ),
        "chunking_label": "Chunking strategy",
        "chunking_help": "Choose how documents are split into text chunks.",
        "topk_label": "Number of sources (top_k)",
        "topk_help": "Number of document fragments retrieved from the search index.",
        "title": "Document Assistant",
        "title_badge": "Demo",
        "subtitle": (
            "A document intelligence system covering PDF ingestion, semantic chunking, "
            "hybrid retrieval with reranking, "
            "and LLM-generated answers with source citations. The LLM layer is provider-agnostic. "
            "Two modes: a fixed pipeline for lightweight models, a LangGraph ReAct agent "
            "for queries that need multiple retrieval steps. "
            "Retrieval quality is evaluated with RAGAS."
        ),
        "search_label": "Ask a question ...",
        "search_placeholder": "E.g.: What are the rules for processing personal data?",
        "search_button": "Search",
        "example_button": "Random question",
        "spinner": "Searching documents ...",
        "status_label": "Processing query ...",
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
        "err_rate_limit": "The API quota is temporarily exhausted. Please wait a moment and try again.",
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
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dokumentassistent",
    page_icon="📄",
    layout="centered",
)

st.markdown('<meta name="robots" content="noindex, nofollow">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Analytics — Umami Cloud
# ---------------------------------------------------------------------------
components.html(
    '<script async src="https://cloud.umami.is/script.js"'
    ' data-website-id="cf6c908e-1236-4406-8c02-88aa7c9a0db2"></script>',
    height=0,
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
        options=["fixed_size", "recursive", "semantic"],
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
st.markdown(
    f'<div class="app-title">'
    f'{t["title"]}'
    f'<span style="font-size:1rem; font-weight:500; color:#FFFFFF; '
    f'background:#901A1E; padding:0.15rem 0.55rem; margin-left:0.6rem; '
    f'vertical-align:middle; letter-spacing:0.05em;">'
    f'{t["title_badge"]}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="app-subtitle">{t["subtitle"]}</div>',
    unsafe_allow_html=True,
)

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

# ---------------------------------------------------------------------------
# Query logic
# ---------------------------------------------------------------------------
if search_clicked and question.strip():
    data: dict = {}
    _sse_error: dict | None = None

    with st.status(t["status_label"], expanded=True) as _status:
        try:
            with requests.post(
                f"{API_BASE}/query/stream",
                json={"question": question.strip(), "top_k": top_k, "strategy": strategy},
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

                    _event = json.loads(_line[6:])
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

                    elif _step == "tool_call":
                        _tool_name = _event.get("tool", "")
                        _tool_query = _event.get("query", "")
                        st.write(
                            (f"Værktøj **{_tool_name}** kaldt: _{_tool_query}_")
                            if lang == "da"
                            else (f"Tool **{_tool_name}** called: _{_tool_query}_")
                        )

                    elif _step == "tool_result":
                        _rc = _event.get("result_count", 0)
                        st.write(
                            (f"Hentet **{_rc}** dokumenter")
                            if lang == "da"
                            else (f"Retrieved **{_rc}** documents")
                        )

                    elif _step == "generate":
                        st.write(
                            "Svar genereret"
                            if lang == "da"
                            else "Answer generated"
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
