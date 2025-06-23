import os
import streamlit as st
import PyPDF2

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Set API key from Streamlit secrets into env variable
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def read_and_textify(files):
    texts = []
    sources = []
    for file in files:
        if file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                texts.append(text)
                sources.append(f"{file.name}_page_{i}")
        elif file.type == "text/plain":
            content = file.read().decode("utf-8")
            texts.append(content)
            sources.append(file.name)
    return texts, sources

st.set_page_config(page_title="Multidoc_QnA", layout="centered")
st.title("Multidoc_QnA")
st.write("---")

uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"])

if uploaded_files:
    st.write(f"Loaded {len(uploaded_files)} files")
    docs, metadatas = read_and_textify(uploaded_files)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_texts(docs, embeddings, metadatas=[{"source": s} for s in metadatas])

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.header("Ask a question about your documents")
    query = st.text_area("Enter your question here")

    if st.button("Get Response"):
        try:
            with st.spinner("Thinking..."):
                response = qa_chain({"question": query}, return_only_outputs=True)
                st.subheader("Answer:")
                st.write(response["answer"])
                st.subheader("Sources:")
                st.write(response["sources"])
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please upload PDF or TXT files to start.")
