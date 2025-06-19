# src/config.py
import streamlit as st

# # === File Paths ===
# NOTES_FILE = f"data/users/{st.session_state["uid"]}/notes.txt"
# FAISS_INDEX_PATH = f"data/users/{st.session_state["uid"]}/vectorstore"

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Ollama Base Model ===
OLLAMA_BASE_MODEL = "llama3.2"

# === Chunking Settings ===
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# === Streamlit UI ===
APP_TITLE = "Assistant"
SYSTEM_PROMPT = (
    "You are a calm, focused cybersecurity and AWS assistant helping prepare for Cyber Fortress."
    "Answer clearly and concisely. If code or CLI is required, show that first."
)
