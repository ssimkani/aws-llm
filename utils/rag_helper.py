# src/rag_helper.py

from google.generativeai.types import GenerationConfig
import time
import pickle
import faiss
import streamlit as st
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.schema import Document as LCDocument
from .config import *
from utils.paths import *
from pathlib import Path


def build_prompt_with_history(user_input, docs, chat_history):
    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
            if msg["role"] in ["user", "assistant"]
        ]
    )

    prompt = f"""You will answer my input based on the 
provided notes, conversation history, and your own knowledge.

Notes:
{context}

Conversation History:
{history_text}

My Input:
{user_input}
"""
    return prompt


def stream_rag_response(user_input, llm, retriever, chat_history, temperature):

    # Retrieve docs
    docs = retriever.invoke(user_input)

    # Build prompt with embedded context
    prompt = build_prompt_with_history(user_input, docs, chat_history)

    response_container = st.empty()
    response = llm.generate_content(prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
                                    ))
    response_text = response.text.strip()

    # streaming output
    with response_container:
        for i in range(1, len(response_text) + 1):
            st.markdown(response_text[:i] + "▌")
            time.sleep(0.001)
        st.markdown(response_text)

    return response_text, docs


# === Load Notes from .txt ===
def load_notes() -> List[Document]:
    if not os.path.exists(get_notes_path()):
        raise FileNotFoundError(f"Notes file not found at: {get_notes_path()}")

    with open(get_notes_path(), "r", encoding="utf-8") as f:
        text = f.read()

    return [Document(page_content=text)]


# === Split into Chunks ===
def split_notes(docs: List[Document]) -> List[LCDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


# === Load Embedding Model ===
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# === Create or Load FAISS Vector Store ===
def create_vector_store(chunks: List[LCDocument]):
    if not chunks:
        # Determine embedding dimension (e.g., 384 for MiniLM)
        dim = 384
        index = faiss.IndexFlatL2(dim)

        # Save the empty FAISS index
        faiss.write_index(index, os.path.join(get_faiss_index_path(), "index.faiss"))

        # Save an empty metadata index.pkl (if required by your loader)
        with open(os.path.join(get_faiss_index_path(), "index.pkl"), "wb") as f:
            pickle.dump(([], {}), f)

        return

    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(get_faiss_index_path())
    return db


def load_vector_store():
    path = Path(get_faiss_index_path())
    index_path = path / "index.faiss"
    metadata_path = path / "index.pkl"

    if not index_path.exists():

        # 1. Create index directory
        path.mkdir(parents=True, exist_ok=True)

        # 2. Create empty FAISS index
        dim = 384  # match your embedding model's output size
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, str(index_path))

        # 3. Create empty metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(([], {}), f)

    try:
        return FAISS.load_local(
            str(path), get_embeddings(), allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()


# === Query the RAG System ===
def query_rag(vector_store, query: str, top_k: int = 4) -> List[LCDocument]:
    return vector_store.similarity_search(query, k=top_k)
