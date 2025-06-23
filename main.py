import os
import streamlit as st
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

st.set_page_config(page_title="Multidoc QnA", layout="centered")
st.title("📄 Multidoc QnA")

def read_text_from_files(files):
    texts = []
    sources = []
    for file in files:
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

    # Use local embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(texts, embeddings, metadatas=[{"source": s} for s in sources])

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Keep ChatOpenAI if model access works, otherwise fallback
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    except Exception:
        st.error("OpenAI chat model not available. Please upgrade your plan or check key.")
        st.stop()

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about your PDFs")
    if query and st.button("Get Answer"):
        with st.spinner("Thinking..."):
            result = qa_chain({"question": query}, return_only_outputs=True)
            st.subheader("Answer:")
            st.write(result["answer"])
            st.subheader("Sources:")
            st.write(result["sources"])
else:
    st.info("Please upload one or more PDF files.")
