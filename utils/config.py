# src/config.py
import streamlit as st
from env.api_keys import *

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Ollama Base Model ===
OLLAMA_BASE_MODEL = "llama3.2"

# === FIREBASE Settings ===
FIREBASE_API_KEY = FIREBASE

# === Chunking Settings ===
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# === Streamlit UI ===
APP_TITLE = "🛡️ Cyber Fortress"
SYSTEM_PROMPT = (
    "You are a calm, focused cybersecurity and AWS assistant helping prepare for Cyber Fortress."
    "Answer clearly and concisely. If code or CLI is required, show that first."
)
