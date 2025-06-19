# utils/paths.py
import os
import streamlit as st


def get_user_id():
    return st.session_state.get("uid")


def get_notes_path():
    user_id = get_user_id()
    if not user_id:
        raise ValueError("User ID not found in session_state")
    return os.path.join("data", "users", user_id, "notes.txt")


def get_faiss_index_path():
    user_id = get_user_id()
    if not user_id:
        raise ValueError("User ID not found in session_state")
    return os.path.join("data", "users", user_id, "vectorstore")
