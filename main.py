import streamlit as st
import os
import pickle

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Setup your OpenAI API key in Streamlit secrets or env var
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Q&A with caching")

st.title("PDF Q&A with Embeddings Cache")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

CACHE_FILE = "embedding_cache.pkl"

if uploaded_file:
    # Load PDF docs
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    # Split text into chunks (to keep embedding size manageable)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(CACHE_FILE):
        # Load cached vectorstore
        with open(CACHE_FILE, "rb") as f:
            vectordb = pickle.load(f)
        st.success("Loaded embeddings from cache.")
    else:
        # Create vectorstore from embeddings
        vectordb = FAISS.from_documents(texts, embeddings)
        # Save cache to file
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(vectordb, f)
        st.success("Created new embeddings and cached.")

    query = st.text_input("Ask a question about the document:")

    if query:
        # Load chat model (gpt-3.5-turbo)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

        # Get relevant docs
        docs = vectordb.similarity_search(query, k=3)

        # Load QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run chain on docs and query
        answer = chain.run(input_documents=docs, question=query)

        st.markdown("### Answer:")
        st.write(answer)
