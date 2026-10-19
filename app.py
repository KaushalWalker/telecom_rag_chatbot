import os
import streamlit as st
import requests, zipfile, io
from rag_chain import build_chain

# ── Environment Setup ────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
api_key = st.secrets["GROQ_API_KEY"]

# ── Sample Questions ─────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

# ── Cache Chain ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_chain():
    # If you want to preload your Chroma store from GitHub release:
    url = st.secrets["github"]["chroma_store_url"]
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("chroma_store")
    return build_chain()

# ── Session State Initialization ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · openai/gpt-oss-120b")
    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

# ── Main Chat Area ───────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption("Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more.")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve input (chat or sidebar button)
question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

# ── Handle New Question ──────────────────────────────────────────────────────
if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    chain = get_chain()
    with st.chat_message("assistant"):
        response = st.write_stream(chain.stream(question))

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
