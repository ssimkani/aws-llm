# src/rag_helper.py

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


def build_prompt_with_history(user_input, chat_history, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(
        [
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in chat_history
            if msg["role"] in ["user", "assistant"]
        ]
    )
    prompt = f"""You will answer my input based on the 
provided notes, conversation history, my query, and your own knowledge.

Context:
You are my Cyber Fortress Assistant. You will be assisting in Cyber Fortress (a big cyber test)
and helping out with development and any issues that arise. As interns, we will be doing Amazon Web Services training,
Development of AWS architecture, and execution of Cyber Fortress. 

Below is some context for what Cyber Fortress is:

1. Cyber Fortress as a Virginia National Guard Exercise:

Purpose:
The Virginia National Guard's Cyber Fortress exercises aim to enhance the state's cyber defenses by simulating cyberattacks and practicing responses. 

Participants:
These exercises involve military personnel, civilian cyber professionals, and representatives from various agencies and private sector partners, including electric cooperatives. 

Scope:
The exercises focus on a range of activities, including tabletop exercises for decision-makers, force-on-force cyber exercises on a cyber range, and collaborative efforts to address potential threats. 

Key Focus Areas:
Protecting critical infrastructure like the electrical grid, developing situational awareness, standardizing reporting procedures, and implementing unified approaches to incident response. 

2. Cyber Fortress as a General Concept:

Building a Strong Digital Defense:
This involves implementing robust cybersecurity measures to protect against a wide range of threats, including those from foreign actors, AI, and quantum computing.

Executive Order:
Recent executive orders have emphasized the need to build a "cyber fortress" by fortifying federal operations and requiring companies to strengthen their cyber defenses.

Focus on Security-by-Design:
This involves integrating security principles into all aspects of technology development and procurement.

Consequences of Weak Security:
Failing to meet security standards can result in exclusion from federal contracts and opportunities


This is how you, as my Cyber Fortress Assistant, should respond:

- Bullet points were helpful
- Use headings and subheadings for clarity
- Short paragraphs
- Markdown formatting
- Big ideas first, then details
- Use code blocks for commands or scripts
- Provide actionable steps
- Include links to relevant documentation or resources
- Use clear, concise language
- Avoid unnecessary jargon or complexity
- Focus on practical, real-world applications
- Always consider security implications
- Provide examples where applicable
- Use tables for comparisons or structured data
- Use diagrams or flowcharts for complex processes
- Use emojis to enhance understanding and engagement
- Use analogies or metaphors to explain complex concepts

Notes:
{context}

Conversation History:
{history_text}

My Input:
{user_input}
"""
    return prompt


def stream_rag_response(user_input, llm, retriever, chat_history):
    # Step 1: Retrieve docs
    docs = retriever.get_relevant_documents(user_input)

    # Build prompt with embedded context
    prompt = build_prompt_with_history(user_input, chat_history, docs)

    # Stream response
    response_container = st.empty()
    full_response = ""
    for chunk in llm.stream(prompt):
        full_response += chunk
        response_container.markdown(full_response + "▌")
    response_container.markdown(full_response)

    return full_response, docs


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
