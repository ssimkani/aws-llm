# src/config.py
import streamlit as st

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Model ===
MODEL = "ssimkani/cf"
MODEL_URL = "http://host.docker.internal:11434"

# === FIREBASE Settings ===
FIREBASE_API_KEY = "AIzaSyCt50iw2KDfpreGa8MeHYdCVzHbbZCcqwk"

# === Chunking Settings ===
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_HISTORY_TOKENS = 1000

# === Streamlit UI ===
APP_TITLE = "🛡️ Cyber Fortress"
