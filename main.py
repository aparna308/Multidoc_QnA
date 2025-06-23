import os
import streamlit as st
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Set API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.set_page_config(page_title="Multidoc QnA", layout="centered")
st.title("📄 Multidoc QnA")

# Read text from uploaded PDFs
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

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(texts, embeddings, metadatas=[{"source": s} for s in sources])

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question about your PDFs")
    if query and st.button("Get Answer"):
        try:
            with st.spinner("Thinking..."):
                result = qa_chain({"question": query}, return_only_outputs=True)
                st.subheader("Answer:")
                st.write(result["answer"])
                st.subheader("Sources:")
                st.write(result["sources"])
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please upload one or more PDF files.")
