import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Load your OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.title("PDF Q&A with OpenAI (Legacy SDK)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        st.error("No extractable text found in the PDF.")
        st.stop()

    # Split text into chunks for embeddings - very naive split here
    texts = [text[i : i + 1000] for i in range(0, len(text), 1000)]

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = vectordb.similarity_search(query, k=4)
        answer = chain.run(input_documents=docs, question=query)
        st.write("**Answer:**", answer)
