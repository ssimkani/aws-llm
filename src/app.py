import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

# ---- SETTINGS ----
TXT_FILE_PATH = "data/aws_notes.txt"
INDEX_DIR = "vectorstore/aws_faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:latest"

st.set_page_config(page_title="AWS Chatbot", layout="wide")
st.title("AWS Chatbot")


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # If FAISS index already exists, load it
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    # Else, create it from the .txt file
    with open(TXT_FILE_PATH, "r") as file:
        raw_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save FAISS index to disk
    vectorstore.save_local(INDEX_DIR)

    return vectorstore


vectorstore = load_vectorstore()

# Setup Ollama LLaMA 3
llm = Ollama(model=LLM_MODEL)

# Retrieval QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)


# ---- STREAMLIT UI ----
user_query = st.text_area("Ask something about AWS:", height=100)

if st.button("Submit") and user_query.strip():
    with st.spinner("Thinking..."):
        result = qa_chain({"query": user_query})

        st.subheader("📘 Answer")
        st.write(result["result"])

        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n---")
