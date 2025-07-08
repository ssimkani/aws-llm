# src/pages/chat.py

import google.generativeai as genai
from utils.firebase_db import save_user_data
import os
import streamlit as st
from utils.rag_helper import load_vector_store, stream_rag_response, create_vector_store
from utils.config import *
from langchain.chains import RetrievalQA
from datetime import datetime
import time

st.set_page_config(page_title=APP_TITLE, layout="wide")

# === User Authentication Check ===
if "uid" not in st.session_state:
    st.warning("Please log in to access chat feature.")
    time.sleep(1)
    st.switch_page("login.py")
    st.stop()

from utils.firebase_db import load_user_notes_text, restore_faiss_files

uid = st.session_state["uid"]
user_dir = f"data/users/{uid}"
vector_path = os.path.join(user_dir, "vectorstore")

# Create local folders if they don't exist
os.makedirs(vector_path, exist_ok=True)

# Restore notes.txt from Firestore
notes_text = load_user_notes_text(uid)
with open(os.path.join(user_dir, "notes.txt"), "w", encoding="utf-8") as f:
    f.write(notes_text)

# Restore FAISS files from Firestore
restore_faiss_files(uid, vector_path)


st.title(APP_TITLE)
st.markdown(
    "<style>" + open("./style/style.css").read() + "</style>", unsafe_allow_html=True
)

# === Session State Initialization ===
if st.session_state.get("reset_chat", False):
    st.session_state["messages"] = []
    st.session_state["reset_chat"] = False

# Ensure messages is initialized if not set or was cleared above
if "messages" not in st.session_state or not st.session_state["messages"]:
    st.session_state["messages"] = []

# === Load Vector Store ===
@st.cache_resource(ttl=30)
def get_vector_store():
    try:
        return load_vector_store()
    except FileNotFoundError:
        
        # Vector store missing
        create_vector_store(chunks=[])
        
        # After creation load again
        return load_vector_store()

vector_store = get_vector_store()


# Add temp to session state
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.3  # preferred default

# === Sidebar: Temperature Slider ===
st.sidebar.markdown("### 🤖 Model Behavior")
st.session_state["temperature"] = st.sidebar.slider(
    "Response Style\n\n(Precision ←→ Creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Lower = deterministic, higher = creative",
)

st.sidebar.markdown(
    f"**Current Behavior:** {'🎯 Precise' if st.session_state["temperature"] < 0.4 else '🧠 Creative' if st.session_state["temperature"] > 0.6 else '⚖️ Balanced'}"
)

# === Load LLM ===
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("models/gemini-1.5-flash")


# Previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat Input
if user_input := st.chat_input("Ask me anything..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # === Call RAG pipeline with Gemini ===
    with st.chat_message("assistant"):
        # Show loading spinner
        with st.spinner("Thinking..."):

            # Call the RAG response function
            response_text, sources = stream_rag_response(
                user_input=user_input,
                llm=llm,
                retriever=vector_store.as_retriever(),
                chat_history=st.session_state.messages,
                temperature=st.session_state["temperature"],
            )

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    # prompt = st.chat_input("Ask me anything...")
    # if prompt:
    #     # Add user message to session state
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    #     if len(st.session_state.messages) >= 3:
    #         history = st.session_state.messages[-3:-1]
    #     else:
    #         history = []

    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         # Stream assistant response
    #         retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    #         response, sources = stream_rag_response(prompt,
    #                                                 llm,
    #                                                 retriever,
    #                                                 history
    #                                                 )
    #         st.session_state.messages.append({"role": "assistant", "content": response})

    # Source documents
    with st.expander("📚 Relevant Notes"):
        for i, doc in enumerate(sources):
            st.markdown(
                f"""
                <div class=\"source-chunk\">
                    <div class=\"chunk-title\">#{i+1}</div>
                    <div class=\"chunk-body\">{doc.page_content.strip()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# === New Chat Button in Container ===

for _ in range(15):
    st.sidebar.write("")

if st.sidebar.button("🆕 New Chat"):
    st.session_state["messages"] = []
    st.rerun()

with st.sidebar:
    if st.button("🔼 Save Notes to Cloud"):
        uid = st.session_state["uid"]
        user_folder = f"data/users/{uid}"
        save_user_data(uid, user_folder)
        st.success("✅ Folder uploaded to Firestore!")
    
    if st.button("🔓 Log Out"):
        for key in ["email", "uid", "id_token"]:
            st.session_state.pop(key, None)
        st.rerun()
