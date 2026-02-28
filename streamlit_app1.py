"""
streamlit_app.py
ShopMind AI — Full RAG + Text-to-SQL chatbot
Updates:
  - Uses improved classifier with word boundary matching
  - Shows classification method + confidence scores in UI
  - Shows ambiguous query warning in UI
  - Typewriter effect for bot responses
  - Classifier debug panel in expander
"""
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from classifier import classify_intent, get_classification_details
from text_to_sql import generate_sql, run_sql, format_answer
import time
import re

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _safe_content(text: str) -> str:
    """
    Strips any HTML tags from text before storing in session state.
    Prevents raw HTML leaking into chat bubbles on re-render.
    """
    return re.sub(r'<[^>]+>', '', str(text)).strip()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161a23;
    --card:      #1e2330;
    --border:    #2a3045;
    --accent:    #f0a500;
    --accent2:   #e05c2a;
    --text:      #e8eaf0;
    --muted:     #7a8099;
    --success:   #2dd4a0;
    --danger:    #f05c5c;
    --warn:      #f0c040;
    --sql-col:   #2dd4a0;
    --rag-col:   #6a8fd8;
    --ambig-col: #f0c040;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1200px; }

/* ── Banner ── */
.top-banner {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 28px;
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid var(--border); border-radius: 16px;
    margin-bottom: 12px; box-shadow: 0 4px 32px rgba(0,0,0,0.4);
}
.top-banner .logo {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.top-banner h1 {
    font-family: 'Syne', sans-serif !important; font-size: 1.7rem !important;
    font-weight: 800 !important; color: var(--text) !important;
    margin: 0 !important; padding: 0 !important; line-height: 1 !important;
}
.top-banner .sub {
    font-size: 0.82rem; color: var(--muted);
    margin-top: 3px; font-weight: 300; letter-spacing: 0.04em;
}
.top-banner .pill {
    margin-left: auto;
    background: rgba(45,212,160,0.12); border: 1px solid rgba(45,212,160,0.3);
    color: var(--success); font-size: 0.75rem; font-weight: 500;
    padding: 5px 14px; border-radius: 20px; letter-spacing: 0.05em; white-space: nowrap;
}

/* ── Status bar ── */
.status-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 10px 20px; margin-bottom: 16px; font-size: 0.82rem;
}
.status-bar .left  { display: flex; align-items: center; gap: 18px; }
.status-bar .tag   { color: var(--muted); }
.status-bar .val   { color: var(--text); font-weight: 500; }
.connected    { color: var(--success); font-weight: 600; }
.disconnected { color: var(--danger);  font-weight: 600; }

/* ── Suggest label ── */
.suggest-label {
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
}

/* ── Chat window ── */
.chat-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px;
    min-height: 420px; max-height: 540px; overflow-y: auto;
    margin-bottom: 16px; scroll-behavior: smooth;
}
.chat-wrap::-webkit-scrollbar { width: 5px; }
.chat-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Message bubbles ── */
.msg-row { display: flex; gap: 12px; margin-bottom: 20px; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.avatar.user { background: linear-gradient(135deg, #3a4a7a, #2a3560); }

.bubble {
    max-width: 72%; padding: 14px 18px; border-radius: 16px;
    font-size: 0.92rem; line-height: 1.65;
}
.bubble.user {
    background: #1a2540;
    border: 1px solid rgba(100,130,220,0.25); border-top-right-radius: 4px;
    color: #c8d0e8;
}
.bubble.bot {
    background: var(--card); border: 1px solid var(--border);
    border-top-left-radius: 4px;
}
.bubble .ts {
    display: block; font-size: 0.67rem; color: var(--muted);
    margin-top: 8px; text-align: right;
}

/* ── Badge row ── */
.badge-row { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }

.badge-sql {
    background: #1a3a2a; color: var(--sql-col);
    font-size: 0.68rem; padding: 2px 10px; border-radius: 10px;
    display: inline-block; font-weight: 600;
}
.badge-rag {
    background: #1a2540; color: var(--rag-col);
    font-size: 0.68rem; padding: 2px 10px; border-radius: 10px;
    display: inline-block; font-weight: 600;
}
.badge-ambig {
    background: #2a2510; color: var(--ambig-col);
    font-size: 0.68rem; padding: 2px 10px; border-radius: 10px;
    display: inline-block; font-weight: 600;
}
.badge-method {
    background: rgba(255,255,255,0.04); color: var(--muted);
    font-size: 0.63rem; padding: 2px 8px; border-radius: 8px;
    display: inline-block;
}
.score-pill {
    font-size: 0.63rem; color: var(--muted);
    padding: 2px 8px; border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    display: inline-block;
}

/* ── Ambiguous warning banner ── */
.ambig-warn {
    background: rgba(240,192,64,0.08); border: 1px solid rgba(240,192,64,0.25);
    border-radius: 10px; padding: 8px 14px; margin-bottom: 10px;
    font-size: 0.78rem; color: var(--ambig-col);
    display: flex; align-items: center; gap: 8px;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 {
    font-family: 'Syne', sans-serif; font-size: 1.1rem;
    color: #4a5270; margin-bottom: 6px;
}

/* ── Doc item ── */
.doc-item {
    padding: 8px 10px; background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;
    margin-bottom: 6px; font-size: 0.78rem; color: #8a91a8; line-height: 1.5;
}
.doc-item b { color: var(--accent); }

/* ── Input ── */
.stTextInput input {
    background: var(--card) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
    padding: 14px 18px !important; transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(240,165,0,0.12) !important;
}
.stTextInput input::placeholder { color: var(--muted) !important; }

/* ── Send button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0d0f14 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    border: none !important; border-radius: 12px !important;
    padding: 12px 24px !important; width: 100%;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Clear button ── */
.clear-btn > button {
    background: transparent !important; color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 400 !important;
}
.clear-btn > button:hover {
    color: var(--danger) !important; border-color: var(--danger) !important;
    background: rgba(240,92,92,0.06) !important;
}

/* ── Suggest buttons ── */
div[data-testid="column"] .stButton > button {
    background: rgba(240,165,0,0.08) !important;
    color: var(--accent) !important;
    border: 1px solid rgba(240,165,0,0.22) !important;
    font-size: 0.73rem !important; font-weight: 500 !important;
    padding: 6px 10px !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(240,165,0,0.18) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

/* ── Spinner text ── */
.stSpinner > div { color: var(--accent) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHROMA_DIR = "db/chroma_db"
MODEL_NAME = "llama3.2:1b"


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "chat_history":     [],
    "display_messages": [],
    "total_queries":    0,
    "sql_queries":      0,
    "rag_queries":      0,
    "ambig_queries":    0,
    "db":               None,
    "model":            None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# LOAD RESOURCES (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources(model_name: str, persist_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    llm = ChatOllama(model=model_name, temperature=0)
    return db, llm


# Auto-load on startup
if st.session_state.db is None:
    try:
        _db, _llm = load_resources(MODEL_NAME, CHROMA_DIR)
        st.session_state.db    = _db
        st.session_state.model = _llm
    except Exception:
        pass


# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(user_question: str) -> tuple[str, list, str, str, dict]:
    """
    Routes question through the correct pipeline.

    Returns:
        answer       : str   — final answer text
        docs         : list  — retrieved docs (semantic only)
        intent       : str   — 'sql' or 'semantic'
        sql_used     : str   — SQL query string (sql only)
        clf_details  : dict  — classification breakdown for UI display
    """
    db    = st.session_state.db
    model = st.session_state.model

    # ── Step 1: Classify intent (improved classifier) ──────────────────
    # get_classification_details gives us full breakdown for UI
    clf_details = get_classification_details(user_question)

    # Final intent — use LLM for ambiguous cases
    intent = classify_intent(user_question, model)
    clf_details["final_intent"] = intent

    # ── SQL Pipeline ───────────────────────────────────────────────────
    if intent == "sql":
        sql     = generate_sql(user_question)
        results = run_sql(sql)
        answer  = format_answer(results, user_question)

        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))
        return answer, [], "sql", sql, clf_details

    # ── Semantic / RAG Pipeline ────────────────────────────────────────
    chat_history = st.session_state.chat_history

    # Rewrite question if history exists
    if chat_history:
        rewrite_msgs = [
            SystemMessage(content=(
                "You are a search query rewriter for a product database. "
                "Rewrite the question as a short 3-6 word search query. "
                "Output ONLY the rewritten query. Nothing else."
            )),
        ] + chat_history + [
            HumanMessage(content=f"Rewrite this question: {user_question}")
        ]
        result   = model.invoke(rewrite_msgs)
        search_q = result.content.strip().split("\n")[0]
    else:
        search_q = user_question

    # Retrieve top 5 docs from ChromaDB
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs      = retriever.invoke(search_q)

    if not docs:
        answer = "❌ No relevant products found for your query."
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))
        return answer, [], "semantic", "", clf_details

    # Build context + answer
    context = "\n".join([f"- {doc.page_content}" for doc in docs])
    prompt  = f"""Answer this question: {user_question}

Products found:
{context}

Rules:
- Answer using ONLY the products listed above
- Be specific: include product_name, final_price, discount_percent, rating, seller when relevant
- Use ₹ for prices
- If the exact product is not found say: "This product is not available in our store."
"""
    answer_msgs = [
        SystemMessage(content=(
            "You are a helpful product assistant for an online store. "
            "Answer concisely using only the provided product data. "
            "Use ₹ for prices. Be friendly and specific."
        )),
        HumanMessage(content=prompt),
    ]
    result = model.invoke(answer_msgs)
    answer = result.content

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    return answer, docs, "semantic", "", clf_details


# ─────────────────────────────────────────────
# TYPEWRITER DISPLAY
# ─────────────────────────────────────────────
def typewriter(text: str, speed: float = 0.025):
    """Displays text word by word for a smoother feel."""
    placeholder = st.empty()
    words       = text.split()
    displayed   = ""
    for word in words:
        displayed += word + " "
        placeholder.markdown(
            f"<div style='font-size:0.92rem;line-height:1.65;color:#e8eaf0'>"
            f"{displayed}▌</div>",
            unsafe_allow_html=True,
        )
        time.sleep(speed)
    placeholder.markdown(
        f"<div style='font-size:0.92rem;line-height:1.65;color:#e8eaf0'>"
        f"{displayed}</div>",
        unsafe_allow_html=True,
    )
    return displayed.strip()


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='top-banner'>
    <div class='logo'>🛍️</div>
    <div>
        <h1>ShopMind AI</h1>
        <div class='sub'>Retrieval-Augmented Product Intelligence · RAG + SQL</div>
    </div>
    <div class='pill'>● LIVE</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — STATUS BAR
# ─────────────────────────────────────────────
is_connected  = st.session_state.db is not None
status_cls    = "connected" if is_connected else "disconnected"
status_text   = "🟢 Connected" if is_connected else "🔴 Not Connected"
products_text = "—"
if is_connected:
    try:
        products_text = f"{st.session_state.db._collection.count():,} products"
    except Exception:
        products_text = "DB ready"

col_stat, col_clr = st.columns([5, 1])
with col_stat:
    st.markdown(f"""
    <div class='status-bar'>
        <div class='left'>
            <span class='{status_cls}'>{status_text}</span>
            <span class='tag'>|</span>
            <span class='val'>📦 {products_text}</span>
            <span class='tag'>|</span>
            <span class='val'>💬 {st.session_state.total_queries} total</span>
            <span class='tag'>|</span>
            <span style='color:#2dd4a0;font-size:0.78rem;font-weight:600'>
                ⚡ {st.session_state.sql_queries} SQL
            </span>
            <span class='tag'>·</span>
            <span style='color:#6a8fd8;font-size:0.78rem;font-weight:600'>
                🔍 {st.session_state.rag_queries} RAG
            </span>
            <span class='tag'>·</span>
            <span style='color:#f0c040;font-size:0.78rem;font-weight:600'>
                ⚠️ {st.session_state.ambig_queries} LLM classified
            </span>
        </div>
        <div style='color:#3a4060;font-size:0.72rem;letter-spacing:0.05em'>SHOPMIND AI</div>
    </div>
    """, unsafe_allow_html=True)

with col_clr:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history     = []
        st.session_state.display_messages = []
        st.session_state.total_queries    = 0
        st.session_state.sql_queries      = 0
        st.session_state.rag_queries      = 0
        st.session_state.ambig_queries    = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — SUGGESTED QUESTIONS
# ─────────────────────────────────────────────
st.markdown("<div class='suggest-label'>💡 Try asking</div>", unsafe_allow_html=True)

suggestions = [
    "How many Adidas products?",
    "Price of Adidas Ultra 664?",
    "Top 5 best rated products?",
    "Cheapest Nike products?",
    "Products with 40% discount?",
    "Average rating by brand?",
    "Out of stock products?",
    "Sort by price low to high?",
]
s_cols = st.columns(len(suggestions))
for i, (col, sug) in enumerate(zip(s_cols, suggestions)):
    with col:
        if st.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_q"] = sug


# ─────────────────────────────────────────────
# UI — CHAT WINDOW
# ─────────────────────────────────────────────

if not st.session_state.display_messages:
    st.markdown("""
    <div class='chat-wrap'>
        <div class='empty-state'>
            <div class='icon'>🔍</div>
            <h3>Start a conversation</h3>
            <p>
                Ask product details →
                <b style='color:#6a8fd8'>RAG Search</b> &nbsp;|&nbsp;
                Ask counts, averages, rankings →
                <b style='color:#2dd4a0'>SQL Query</b>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.display_messages):
        role      = msg["role"]
        intent    = msg.get("intent", "")
        sql       = msg.get("sql", "")
        clf       = msg.get("clf_details", {})
        method    = clf.get("method", "")
        sql_score = clf.get("sql_score", 0)
        sem_score = clf.get("semantic_score", 0)
        was_ambig = method == "llm_fallback" or clf.get("intent") == "ambiguous"
        content   = msg["content"]
        ts        = msg.get("ts", "")

        # ── USER message ─────────────────────────────────────────────────
        if role == "user":
            # layout: empty col left | content col | avatar col
            _, col_content, col_av = st.columns([1, 6, 0.5])
            with col_av:
                st.markdown(
                    "<div class='avatar user' style='margin-top:6px'>👤</div>",
                    unsafe_allow_html=True,
                )
            with col_content:
                st.markdown(
                    f"""<div class='bubble user'>
                            {content}
                            <span class='ts'>{ts}</span>
                        </div>""",
                    unsafe_allow_html=True,
                )

        # ── ASSISTANT message ─────────────────────────────────────────────
        else:
            # layout: avatar col | content col | empty col right
            col_av, col_content, _ = st.columns([0.5, 6, 1])

            with col_av:
                st.markdown(
                    "<div class='avatar bot' style='margin-top:6px'>🤖</div>",
                    unsafe_allow_html=True,
                )

            with col_content:
                # ── Badge ──
                if intent == "sql":
                    st.markdown(
                        f"<div class='badge-row'>"
                        f"<span class='badge-sql'>⚡ SQL Query</span>"
                        f"<span class='badge-method'>{method}</span>"
                        f"<span class='score-pill'>SQL:{sql_score} · SEM:{sem_score}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                elif intent == "semantic":
                    st.markdown(
                        f"<div class='badge-row'>"
                        f"<span class='badge-rag'>🔍 RAG Search</span>"
                        f"<span class='badge-method'>{method}</span>"
                        f"<span class='score-pill'>SQL:{sql_score} · SEM:{sem_score}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Ambiguous warning ──
                if was_ambig:
                    st.markdown(
                        "<div class='ambig-warn'>"
                        "⚠️ Query was ambiguous — classified by LLM fallback"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Answer content bubble ──
                # Use st.markdown natively so bold/formatting renders correctly
                st.markdown(
                    f"<div class='bubble bot'>{content}"
                    f"<span class='ts'>{ts}</span></div>",
                    unsafe_allow_html=True,
                )

                # ── SQL expander ──
                if sql:
                    with st.expander("🗄️ SQL Query Used", expanded=False):
                        st.code(sql, language="sql")

                # ── Classifier debug expander ──
                if clf:
                    with st.expander("🧠 Classifier Details", expanded=False):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Intent",    clf.get("final_intent", intent).upper())
                        c2.metric("SQL Score", clf.get("sql_score", 0))
                        c3.metric("SEM Score", clf.get("semantic_score", 0))
                        c4.metric("Method",    clf.get("method", "—"))
                        if clf.get("product_pattern"):
                            st.info("🏷️ Product name pattern detected → routed to RAG")
                        if clf.get("sql_override"):
                            st.warning("⚡ SQL override keyword present")

                # ── RAG docs expander ──
                if msg.get("docs"):
                    with st.expander(
                        f"📄 {len(msg['docs'])} source documents retrieved",
                        expanded=False,
                    ):
                        for j, doc in enumerate(msg["docs"], 1):
                            name       = doc.page_content.split(".")[0]
                            price_part = ""
                            for part in doc.page_content.split("."):
                                if "Final price" in part:
                                    price_part = part.strip()
                                    break
                            st.markdown(
                                f"<div class='doc-item'>"
                                f"<b>#{j} — {name}</b>"
                                f"{'<br>' + price_part if price_part else ''}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — INPUT ROW
# ─────────────────────────────────────────────
col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input = st.text_input(
        "q",
        value=st.session_state.pop("pending_q", ""),
        placeholder="Ask about any product, price, count, rating, discount...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)


# ─────────────────────────────────────────────
# HANDLE SEND
# ─────────────────────────────────────────────
if send and user_input.strip():
    if not st.session_state.db:
        st.error("⚠️ Database not loaded. Make sure db/chroma_db exists and run 1_ingestion_pipeline.py first.")
    else:
        question = user_input.strip()
        ts       = time.strftime("%H:%M")

        # Add user message — _safe_content strips any accidental HTML tags
        st.session_state.display_messages.append({
            "role":    "user",
            "content": _safe_content(question),
            "ts":      ts,
        })

        # Run pipeline
        with st.spinner("🔍 Thinking..."):
            try:
                answer, docs, intent, sql, clf_details = run_pipeline(question)

                # Update session counters
                st.session_state.total_queries += 1
                if intent == "sql":
                    st.session_state.sql_queries += 1
                else:
                    st.session_state.rag_queries += 1
                if clf_details.get("method") == "llm_fallback":
                    st.session_state.ambig_queries += 1

                st.session_state.display_messages.append({
                    "role":        "assistant",
                    "content":     _safe_content(answer),
                    "docs":        docs,
                    "intent":      intent,
                    "sql":         sql,
                    "clf_details": clf_details,
                    "ts":          time.strftime("%H:%M"),
                })

            except Exception as e:
                st.session_state.display_messages.append({
                    "role":        "assistant",
                    "content":     _safe_content(f"⚠️ Error: {str(e)}"),
                    "docs":        [],
                    "intent":      "",
                    "sql":         "",
                    "clf_details": {},
                    "ts":          time.strftime("%H:%M"),
                })

        st.rerun()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:16px 0 4px 0;
    font-size:0.72rem; color:#3a4060; letter-spacing:0.06em'>
    SHOPMIND AI &nbsp;·&nbsp; LANGCHAIN &nbsp;·&nbsp; CHROMADB
    &nbsp;·&nbsp; SQLITE &nbsp;·&nbsp; OLLAMA
</div>
""", unsafe_allow_html=True)