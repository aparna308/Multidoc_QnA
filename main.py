import os
import streamlit as st
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# 1) Set API key—**must match exactly** 'openai_api_key' in Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def read_pdf(files):
    texts, sources = [], []
    for file in files:
        pdf = PyPDF2.PdfReader(file)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            texts.append(text)
            sources.append(f"{file.name}_page_{i}")
    return texts, sources

st.set_page_config(page_title="Multidoc Q&A", layout="centered")
st.title("Multidoc Q&A")

uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
if uploaded:
    st.write(f"📄 Loaded {len(uploaded)} PDF(s).")
    docs, srcs = read_pdf(uploaded)

    # 2) Create embeddings with default constructor (langchain==0.0.145)
    embeddings = OpenAIEmbeddings()

    # 3) Build vector store
    vectordb = Chroma.from_texts(docs, embeddings, metadatas=[{"source": s} for s in srcs])
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.text_input("Ask a question about your document:")
    if query and st.button("Get Answer"):
        with st.spinner("Thinking..."):
            try:
                ans = qa({"question": query})
                st.subheader("Answer:")
                st.write(ans["answer"])
                st.subheader("Sources:")
                st.write(ans["sources"])
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload at least one PDF to get started.")
