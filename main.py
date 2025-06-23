import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    st.title("PDF Q&A with OpenAI & Langchain")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        # Load documents from PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Initialize embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(docs, embeddings)

        # Initialize Chat model
        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Setup retrieval QA chain
        qa = RetrievalQA.from_chain_type(llm=chat, retriever=vectordb.as_retriever())

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                answer = qa.run(query)
            st.write("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
