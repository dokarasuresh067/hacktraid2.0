import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShopMind AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark luxury theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0d0f14;
    --surface:   #161a23;
    --card:      #1e2330;
    --border:    #2a3045;
    --accent:    #f0a500;
    --accent2:   #e05c2a;
    --text:      #e8eaf0;
    --muted:     #7a8099;
    --user-bg:   #1a2540;
    --bot-bg:    #1e2330;
    --success:   #2dd4a0;
    --danger:    #f05c5c;
}

/* ── Global ── */
html, body, [class*="css"]  {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1200px; }

/* ── Top Header Banner ── */
.top-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 28px;
    background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
}
.top-banner .logo-circle {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.top-banner h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    margin: 0 !important; padding: 0 !important;
    line-height: 1 !important;
}
.top-banner .subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 3px;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.top-banner .status-pill {
    margin-left: auto;
    background: rgba(45, 212, 160, 0.12);
    border: 1px solid rgba(45, 212, 160, 0.3);
    color: var(--success);
    font-size: 0.75rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px !important;
}
.sidebar-stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.sidebar-stat-card .label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.sidebar-stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
}
.sidebar-stat-card .value.green { color: var(--success); }
.sidebar-stat-card .value.small {
    font-size: 0.9rem;
    color: var(--text);
    font-weight: 500;
}

/* ── Suggested Questions ── */
.suggest-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

/* ── Chat Container ── */
.chat-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    min-height: 420px;
    max-height: 560px;
    overflow-y: auto;
    margin-bottom: 16px;
    scroll-behavior: smooth;
}
.chat-wrap::-webkit-scrollbar { width: 5px; }
.chat-wrap::-webkit-scrollbar-track { background: transparent; }
.chat-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Message Bubbles ── */
.msg-row { display: flex; gap: 12px; margin-bottom: 20px; align-items: flex-start; }
.msg-row.user  { flex-direction: row-reverse; }
.avatar {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.avatar.user { background: linear-gradient(135deg, #3a4a7a, #2a3560); }
.bubble {
    max-width: 72%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.65;
    position: relative;
}
.bubble.user {
    background: var(--user-bg);
    border: 1px solid rgba(100,130,220,0.25);
    border-top-right-radius: 4px;
    color: #c8d0e8;
}
.bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
}
.bubble .ts {
    display: block;
    font-size: 0.67rem;
    color: var(--muted);
    margin-top: 8px;
    text-align: right;
}

/* ── Source Docs Expander inside bubble ── */
.doc-chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.doc-chip {
    background: rgba(240,165,0,0.08);
    border: 1px solid rgba(240,165,0,0.22);
    color: var(--accent);
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #4a5270;
    margin-bottom: 6px;
}
.empty-state p { font-size: 0.85rem; }

/* ── Input Row ── */
.stTextInput input {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(240,165,0,0.12) !important;
}
.stTextInput input::placeholder { color: var(--muted) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.03em;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* secondary clear button */
.clear-btn > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
}
.clear-btn > button:hover {
    color: var(--danger) !important;
    border-color: var(--danger) !important;
    background: rgba(240,92,92,0.06) !important;
}

/* ── Thinking indicator ── */
.thinking {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 16px;
}
.dot-pulse {
    display: flex; gap: 4px;
}
.dot-pulse span {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.2s ease-in-out infinite;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%            { opacity: 1;   transform: scale(1.1); }
}

/* ── Retrieved docs section ── */
.docs-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 10px;
    font-size: 0.8rem;
}
.docs-section .doc-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 8px;
}
.doc-item {
    padding: 8px 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: #8a91a8;
    line-height: 1.5;
}
.doc-item b { color: var(--accent); font-weight: 500; }

/* ── selectbox / radio ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # LangChain message objects
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []      # {role, content, docs, ts}
if "db" not in st.session_state:
    st.session_state.db = None
if "model" not in st.session_state:
    st.session_state.model = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []


# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources(model_name: str, persist_dir: str):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    model = ChatOllama(model=model_name, temperature=0)
    return db, model


# ─────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────
def classify_intent(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["how many", "count", "total number", "number of"]):
        return "count"
    if any(w in q for w in ["is there", "do you have", "any product", "available", "exist"]):
        return "existence"
    return "semantic"


# ─────────────────────────────────────────────
# CORE RAG LOGIC
# ─────────────────────────────────────────────
def ask_question(user_question: str, db, model) -> tuple[str, list]:
    chat_history = st.session_state.chat_history

    # Step 1 – rewrite if there's history
    if chat_history:
        rewrite_messages = [
            SystemMessage(content=(
                "You are a search query rewriter for a product database. "
                "Rewrite the question as a short 3-6 word search query. "
                "Output ONLY the rewritten query. Nothing else. No explanations."
            )),
        ] + chat_history + [
            HumanMessage(content=f"Rewrite this question: {user_question}")
        ]
        result = model.invoke(rewrite_messages)
        search_question = result.content.strip().split("\n")[0]
    else:
        search_question = user_question

    # Step 2 – retrieve
    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    if not docs:
        answer = "❌ No relevant products found in the database for your query."
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))
        return answer, []

    # Step 3 – build prompt
    combined_input = f"""Answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Answer directly using only the documents above. Be specific — include product names, prices, discounts, ratings etc. when relevant. If the exact product is not found, say "This product is not available in our store."
"""

    # Step 4 – answer (no chat history — docs provide full context)
    answer_messages = [
        SystemMessage(content=(
            "You are a helpful product assistant for an online store. "
            "Answer using ONLY the provided product documents. "
            "Be concise, friendly, and specific. Use ₹ for prices. "
            "Format your answer in clean readable text."
        )),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(answer_messages)
    answer = result.content

    # Step 5 – save history
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    return answer, docs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0'>
        <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 800; color: #e8eaf0; margin-bottom: 4px'>⚙️ Configuration</div>
        <div style='font-size: 0.75rem; color: #7a8099'>Connect your RAG backend</div>
    </div>
    """, unsafe_allow_html=True)

    ### Model config
    st.markdown("### 🤖 Model")
    model_name = st.selectbox(
        "Ollama Model",
        ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b", "mistral:7b", "gemma2:2b"],
        label_visibility="collapsed",
    )

    st.markdown("### 📁 Vector Store")
    persist_dir = st.text_input(
        "ChromaDB Path",
        value="db/chroma_db",
        label_visibility="collapsed",
        placeholder="db/chroma_db",
    )

    connect_clicked = st.button("🔌 Connect", use_container_width=True)

    if connect_clicked:
        with st.spinner("Loading models & vector store..."):
            try:
                db, llm = load_resources(model_name, persist_dir)
                st.session_state.db = db
                st.session_state.model = llm
                st.success("Connected!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    ### Stats
    st.markdown("### 📊 Session Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='sidebar-stat-card'>
            <div class='label'>Queries</div>
            <div class='value'>{st.session_state.total_queries}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        turns = len(st.session_state.display_messages)
        st.markdown(f"""
        <div class='sidebar-stat-card'>
            <div class='label'>Messages</div>
            <div class='value green'>{turns}</div>
        </div>""", unsafe_allow_html=True)

    db_status = "✅ Connected" if st.session_state.db else "⚪ Not Connected"
    st.markdown(f"""
    <div class='sidebar-stat-card'>
        <div class='label'>DB Status</div>
        <div class='value small'>{db_status}</div>
    </div>""", unsafe_allow_html=True)

    if st.session_state.db:
        try:
            count = st.session_state.db._collection.count()
            st.markdown(f"""
            <div class='sidebar-stat-card'>
                <div class='label'>Products Indexed</div>
                <div class='value'>{count:,}</div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass

    st.divider()

    ### Clear chat
    st.markdown("### 🗑️ Actions")
    with st.container():
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.display_messages = []
            st.session_state.total_queries = 0
            st.session_state.last_docs = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div class='top-banner'>
    <div class='logo-circle'>🛍️</div>
    <div>
        <h1>ShopMind AI</h1>
        <div class='subtitle'>Retrieval-Augmented Product Intelligence</div>
    </div>
    <div class='status-pill'>● RAG POWERED</div>
</div>
""", unsafe_allow_html=True)

# ── Suggested questions ──
st.markdown("<div class='suggest-label'>💡 Try asking</div>", unsafe_allow_html=True)
suggestions = [
    "What is the price of Adidas Ultra 664?",
    "Show me Dell products under ₹30,000",
    "Which products have 50% discount?",
    "Best rated products available?",
    "Is there any Nike product in stock?",
]
s_cols = st.columns(len(suggestions))
for i, (col, sug) in enumerate(zip(s_cols, suggestions)):
    with col:
        if st.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["pending_question"] = sug

# ── Chat Display ──
st.markdown("<div class='chat-wrap' id='chat-wrap'>", unsafe_allow_html=True)

if not st.session_state.display_messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>🔍</div>
        <h3>Start a conversation</h3>
        <p>Ask about products, prices, availability, brands and more.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.display_messages:
        role   = msg["role"]
        avatar = "🤖" if role == "assistant" else "👤"
        cls    = "bot" if role == "assistant" else "user"
        ts     = msg.get("ts", "")

        bubble_html = f"""
        <div class='msg-row {cls}'>
            <div class='avatar {cls}'>{avatar}</div>
            <div class='bubble {cls}'>
                {msg['content']}
                <span class='ts'>{ts}</span>
            </div>
        </div>
        """
        st.markdown(bubble_html, unsafe_allow_html=True)

        # Show retrieved docs under assistant messages
        if role == "assistant" and msg.get("docs"):
            with st.expander(f"📄 {len(msg['docs'])} source documents retrieved", expanded=False):
                for j, doc in enumerate(msg["docs"], 1):
                    lines = doc.page_content.split(".")
                    name_line = lines[0] if lines else doc.page_content[:80]
                    price_part = ""
                    for part in lines:
                        if "Final price" in part:
                            price_part = part.strip()
                            break
                    st.markdown(f"""
                    <div class='doc-item'>
                        <b>#{j} — {name_line}</b><br>
                        {price_part}
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Input Row ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "question",
        value=st.session_state.pop("pending_question", ""),
        placeholder="Ask about any product, brand, price, availability...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send ➤", use_container_width=True)


# ─────────────────────────────────────────────
# HANDLE SEND
# ─────────────────────────────────────────────
if send and user_input.strip():
    if not st.session_state.db or not st.session_state.model:
        st.error("⚠️ Please connect to the vector store first using the sidebar.")
    else:
        question = user_input.strip()
        ts = time.strftime("%H:%M")

        # Add user message to display
        st.session_state.display_messages.append({
            "role": "user",
            "content": question,
            "ts": ts,
        })

        # Run RAG
        with st.spinner("🔍 Searching products..."):
            try:
                answer, docs = ask_question(
                    question,
                    st.session_state.db,
                    st.session_state.model,
                )
                st.session_state.total_queries += 1
                st.session_state.last_docs = docs

                st.session_state.display_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "docs": docs,
                    "ts": time.strftime("%H:%M"),
                })
            except Exception as e:
                st.session_state.display_messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "docs": [],
                    "ts": time.strftime("%H:%M"),
                })

        st.rerun()

# ── Footer ──
st.markdown("""
<div style='text-align:center; padding: 20px 0 4px 0; font-size: 0.72rem; color: #3a4060; letter-spacing: 0.06em;'>
    SHOPMIND AI  ·  POWERED BY LANGCHAIN + CHROMADB + OLLAMA
</div>
""", unsafe_allow_html=True)