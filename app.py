# streamlit_app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📘",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

h1, h2, h3 {
    color: #f8fafc;
}

.stTextInput > div > div > input {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
    border: 1px solid #334155;
    padding: 12px;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
}

.stButton button {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #06b6d4, #7c3aed);
}

.chat-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid #334155;
}

.user-msg {
    background-color: #2563eb;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
}

.bot-msg {
    background-color: #334155;
    padding: 12px;
    border-radius: 10px;
    color: white;
}

.upload-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

# =========================
# TITLE
# =========================
st.title("📘 RAG PDF Chatbot")
st.markdown("### Chat with your PDF using AI")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload a PDF and ask questions from it.")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# =========================
# PROCESS PDF
# =========================
if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded pdf temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        # Retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        # LLM
        llm = ChatMistralAI(model="mistral-small-2506")

        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant.
                    Use ONLY the provided context to answer the question.
                    If the answer is not present in the context, say:
                    "Could not find the answer in the document."
                    """
                ),
                (
                    "human",
                    "Context: {context}\n\nQuestion: {question}"
                )
            ]
        )

    st.success("✅ PDF Processed Successfully!")

    # =========================
    # QUESTION INPUT
    # =========================
    query = st.text_input("💬 Ask a question from the PDF")

    # =========================
    # ASK BUTTON
    # =========================
    if st.button("Generate Answer"):

        if query.strip() != "":

            with st.spinner("Generating Answer..."):

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.format_prompt(
                    context=context,
                    question=query
                )

                answer = llm.invoke(final_prompt)

                # =========================
                # DISPLAY CHAT
                # =========================
                st.markdown('<div class="chat-box">', unsafe_allow_html=True)

                st.markdown(
                    f'<div class="user-msg"><b>You:</b> {query}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="bot-msg"><b>AI:</b><br>{answer.content}</div>',
                    unsafe_allow_html=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📄 Please upload a PDF to start chatting.")