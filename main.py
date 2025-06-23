import os
import streamlit as st
import PyPDF2

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Set API key from Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.set_page_config(page_title="Multidoc QnA", layout="centered")
st.title("📄 Multidoc QnA (Free-Tier Compatible)")

def read_text_from_files(files):
    texts, sources = [], []
    for file in files:
        if file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    texts.append(text)
                    sources.append(f"{file.name}_page_{i}")
    return texts, sources

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} files")
    texts, sources = read_text_from_files(uploaded_files)

    # Use free-tier embeddings via HF
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.from_texts(texts, embeddings, metadatas=[{"source": s} for s in sources])

    retriever = faiss_index.as_retriever(search_kwargs={"k": 2})

    # Use text-davinci-003 (completion model) on free tier
    llm = OpenAI(model_name="text-davinci-003", temperature=0)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about your PDFs")
    if query and st.button("Get Answer"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"question": query}, return_only_outputs=True)
                st.subheader("Answer:")
                st.write(result["answer"])
                st.subheader("Sources:")
                st.write(result["sources"])
            except Exception as e:
                st.error(f"Error generating answer: {e}")
else:
    st.info("Please upload one or more PDF files to begin.")
