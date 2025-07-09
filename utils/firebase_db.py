# firebase_db.py
import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st

# Load service account key only once
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_cred.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def save_user_data(uid: str, user_folder: str):
    # Save notes.txt
    notes_path = os.path.join(user_folder, "notes.txt")
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            notes_text = f.read()
        db.collection("users").document(uid).collection("notes").document(
            "notes_txt"
        ).set({"text": notes_text, "timestamp": firestore.SERVER_TIMESTAMP})

    # === Save FAISS files
    vectorstore_path = os.path.join(user_folder, "vectorstore")
    for filename in ["index.faiss", "index.pkl"]:
        filepath = os.path.join(vectorstore_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            db.collection("users").document(uid).collection("vectorstore").document(
                filename
            ).set({"content_b64": encoded, "timestamp": firestore.SERVER_TIMESTAMP})


def load_user_notes_text(uid: str):
    doc = (
        db.collection("users")
        .document(uid)
        .collection("notes")
        .document("notes_txt")
        .get()
    )
    data = doc.to_dict()
    return data["text"] if doc.exists else ""


def load_faiss_files(uid: str, destination_path: str):
    vector_ref = db.collection("users").document(uid).collection("vectorstore")
    for filename in ["index.faiss", "index.pkl"]:
        doc = vector_ref.document(filename).get()
        if doc.exists:
            encoded = doc.to_dict()["content_b64"]
            with open(os.path.join(destination_path, filename), "wb") as f:
                f.write(base64.b64decode(encoded))
