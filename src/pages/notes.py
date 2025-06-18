# pages/edit_notes.py

import streamlit as st
from utils.config import NOTES_FILE
from utils.rag_helper import load_notes, split_notes, create_vector_store
from streamlit_ace import st_ace
import time

st.set_page_config(page_title="Notes", layout="wide")

st.title("📝 Notes")
st.markdown(
    "<style>" + open("style/style_notes.css").read() + "</style>", unsafe_allow_html=True
)

with open(NOTES_FILE, "r", encoding="utf-8") as f:
    notes_text = f.read()

# === Code Editor with Line Numbers ===
updated_notes = st_ace(
    value=notes_text,
    language="text",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    tab_size=4,
    show_gutter=True,
    wrap=True,
    auto_update=True,
    height=400,
    key="ace-editor",
)

if st.button("💾 Save"):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(updated_notes)
    st.success("Notes saved.")

    # Trigger rebuild
    from utils.rag_helper import load_notes, split_notes, create_vector_store

    chunks = split_notes(load_notes())
    create_vector_store(chunks)
    st.success("Vector store updated successfully.")

    st.session_state["reset_chat"] = True
    time.sleep(2)
    st.rerun()
