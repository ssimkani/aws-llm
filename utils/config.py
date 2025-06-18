# src/config.py

# === File Paths ===
NOTES_FILE = "data/notes.txt"
FAISS_INDEX_PATH = "vectorstore"
HISTORY_FILE = "history/chat_history.json"

# === Embedding Model ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Ollama Base Model ===
OLLAMA_BASE_MODEL = "llama3.2"

# === Chunking Settings ===
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# === Streamlit UI ===
APP_TITLE = "CF Assistant"
SYSTEM_PROMPT = (
    "You are a calm, focused cybersecurity and AWS assistant helping prepare for Cyber Fortress."
    "Answer clearly and concisely. If code or CLI is required, show that first."
)
