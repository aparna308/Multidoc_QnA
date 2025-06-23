import streamlit as st
import os
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Set API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def read_pdf(files):
    texts = []
    sources = []
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                sources.append(f"{file.name}_page_{i}")
    return texts, sources

st.set_page_config(layout="centered")
st.title("📄 Multidoc Q&A")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} file(s)")

    texts, sources = read_pdf(uploaded_files)

    # Load embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectordb = Chroma.from_texts(texts, embeddings, metadatas=[{"source": s} for s in sources])
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Load LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about the documents")

    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"question": query})
                st.subheader("Answer:")
                st.write(result['answer'])

                st.subheader("Sources:")
                st.write(result['sources'])
            except Exception as e:
                st.error(f"Error: {e}")
