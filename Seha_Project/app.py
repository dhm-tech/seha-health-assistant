import streamlit as st
import torch
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# 🔧 DEVICE CHECK
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Seha RAG System", layout="wide")

st.sidebar.title("🛠 System Status")
if device == "cuda":
    st.sidebar.success("🚀 Running on GPU")
else:
    st.sidebar.warning("⚠️ Running on CPU")

# ==============================
# 🧠 LOAD DATA
# ==============================
@st.cache_resource
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "DATA")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA directory not found at: {data_dir}\n"
            "Make sure the DATA folder is next to app.py."
        )

    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()
    if not docs:
        raise ValueError(
            f"No PDF documents were loaded from {data_dir}.\n"
            "Please add valid PDF files to the DATA directory."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "No text chunks were created from your documents.\n"
            "Check that the PDFs contain selectable text and are not scanned images."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    try:
        db = FAISS.from_documents(chunks, embeddings)
    except IndexError as exc:
        raise ValueError(
            "Embeddings could not be generated from the documents.\n"
            "Verify that DATA/ contains valid readable PDFs and that the model can extract embeddings."
        ) from exc

    return db

with st.spinner("📚 Loading knowledge base..."):
    db = load_data()

st.success("✅ Knowledge Base Ready")

# ==============================
# 🤖 LOAD FLAN-T5 (FIXED)
# ==============================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# 🎯 UI
# ==============================
st.title("🩺 AI Health Assistant (RAG System)")
st.info("Ask about symptoms and get medical guidance.")

query = st.text_input("Enter your symptoms or question:")

# ==============================
# 🔍 PROCESS QUERY
# ==============================
if query:
    with st.spinner("🔍 Analyzing..."):

        # 1. RETRIEVE
        results = db.similarity_search(query, k=5)
        context = " ".join([doc.page_content for doc in results])

        # 2. PROMPT (IMPORTANT)
        prompt = f"""
You are a medical assistant.

Answer ONLY using the provided context.
Do NOT invent information.

Structure your answer EXACTLY like this:

Possible condition:
...

Recommended actions:
...

When to see a doctor:
...

Context:
{context}

Question:
{query}

Answer:
"""

        # 3. GENERATE (FIXED)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        outputs = model.generate(
            **inputs,
            max_length=256
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ==============================
    # 🧠 DISPLAY ANSWER
    # ==============================
    st.subheader("🧠 AI Medical Response")
    st.write(answer)

    # ==============================
    # 📄 SHOW SOURCES
    # ==============================
    st.subheader("📄 Retrieved References")
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "Unknown")
        with st.expander(f"Reference {i+1} - {source}"):
            st.write(doc.page_content)

# ==============================
# ⚠️ DISCLAIMER
# ==============================
st.markdown("---")
st.warning("⚠️ This system provides general guidance only. Consult a doctor for professional advice.")