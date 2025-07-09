import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import os
import time
from utils.firebase_auth import firebase_login, firebase_signup
from utils.firebase_db import *
from utils.paths import *
from utils.config import *
from utils.rag_helper import create_vector_store


st.set_page_config(page_title=APP_TITLE, layout="wide")

# Redirect to chat page if user is already logged in
if "email" in st.session_state and "uid" in st.session_state:
    st.info("You are already logged in. Redirecting to chat page...")
    time.sleep(1)
    st.switch_page("pages/chat.py")

st.title("🛡️ Cyber Fortress Login")

mode = st.radio("Choose mode", ["Login", "Sign Up"], horizontal=True)
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Submit"):
    if mode == "Login":
        result = firebase_login(email, password)
    else:
        result = firebase_signup(email, password)

    if "error" in result:
        st.error(f"{mode} failed: {result['error']['message']}")
    else:
        # Store session info
        st.session_state["email"] = result["email"]
        st.session_state["uid"] = result["localId"]
        st.session_state["id_token"] = result["idToken"]

        user_id = result["localId"]
        user_dir = f"data/users/{user_id}"
        os.makedirs(user_dir, exist_ok=True)

        st.session_state["notes_path"] = f"{user_dir}/notes.txt"
        st.session_state["vector_path"] = f"{user_dir}/vectorstore/"

        # Ensure notes.txt exists
        if not os.path.exists(st.session_state["notes_path"]):
            with open(st.session_state["notes_path"], "w", encoding="utf-8") as f:
                f.write("")

        # Ensure vector store directory exists
        os.makedirs(st.session_state["vector_path"], exist_ok=True)

        # Load from cloud store
        user_dir = f"data/users/{user_id}"
        vector_path = os.path.join(user_dir, "vectorstore")
        # Load notes
        try:
            notes_text = load_user_notes_text(user_id)
            if notes_text:
                with open(os.path.join(user_dir, "notes.txt"), "w", encoding="utf-8") as f:
                    f.write(notes_text)
                st.info("📝 Notes loaded.")
            else:
                st.warning("⚠️ No notes found in Firestore.")
        except Exception as e:
            st.error(f"Failed to load notes: {e}")

        # load FAISS files
        try:
            load_faiss_files(user_id, vector_path)
            st.info("📦 FAISS files loaded.")
        except Exception as e:
            st.error(f"Failed to restore FAISS files: {e}")

        st.success(f"{mode} successful! Welcome, {result['email']}")
        time.sleep(1)

        st.switch_page("pages/chat.py")
