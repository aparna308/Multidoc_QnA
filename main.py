import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    st.title("PDF Q&A with OpenAI & Langchain")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # IMPORTANT: Do NOT pass openai_api_key here,
        # make sure OPENAI_API_KEY is set in your environment before running this app.
        embeddings = OpenAIEmbeddings()

        vectordb = FAISS.from_documents(docs, embeddings)

        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        qa = RetrievalQA.from_chain_type(llm=chat, retriever=vectordb.as_retriever())

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                answer = qa.run(query)
            st.write("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
