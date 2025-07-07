# src/config.py
import streamlit as st

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Model ===
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# === FIREBASE Settings ===
FIREBASE_API_KEY = st.secrets["FIREBASE_API_KEY"]

# === Chunking Settings ===
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# === Streamlit UI ===
APP_TITLE = "🛡️ Cyber Fortress"
