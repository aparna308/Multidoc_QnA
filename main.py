import os
import streamlit as st
import PyPDF2

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Load OpenAI key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.title("📄 Multidoc QnA (Free Tier Compatible)")

def read_text_from_pdfs(files):
    texts, sources = [], []
    for f in files:
        pdf = PyPDF2.PdfReader(f)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                sources.append(f"{f.name}_page_{i}")
    return texts, sources

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} file(s)")
    docs, metadatas = read_text_from_pdfs(uploaded_files)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(docs, embeddings, metadatas=[{"source": m} for m in metadatas])
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = OpenAI(model_name="text-curie-001", temperature=0)
    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Generating answer..."):
            try:
                result = qa({"question": query}, return_only_outputs=True)
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Source pages")
                st.write(result["sources"])
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.info("Upload one or more PDF files to get started.")
