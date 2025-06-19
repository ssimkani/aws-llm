import streamlit as st
import os
import time
from utils.firebase_auth import firebase_login, firebase_signup
from utils.paths import *
from utils.config import *
from utils.rag_helper import create_vector_store


st.set_page_config(page_title=APP_TITLE, layout="wide")

# Redirect to chat page if user is already logged in
if "email" in st.session_state and "uid" in st.session_state:
    st.info("You are already logged in. Redirecting to chat page...")
    time.sleep(1)
    st.switch_page("pages/chat.py")

st.title("🛡️ Cyber Fortress Assistant Login")

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

        st.success(f"{mode} successful! Welcome, {result['email']}")
        time.sleep(1)

        st.switch_page("pages/chat.py")
