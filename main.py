import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2
import os

# Set OpenAI API key environment variable from Streamlit secrets
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

st.set_page_config(layout="centered", page_title="Multidoc_QnA")
st.header("Multidoc_QnA")
st.write("---")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt"])
st.write("---")

if not uploaded_files:
    st.info("Upload files to analyse")
else:
    st.write(f"{len(uploaded_files)} document(s) loaded..")

    documents, sources = read_and_textify(uploaded_files)

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env

    vectordb = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.header("Ask your data")
    user_query = st.text_area("Enter your questions here")

    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = qa_chain({"question": user_query}, return_only_outputs=True)
                st.subheader("Your response:")
                st.write(result["answer"])
                st.subheader("Source pages:")
                st.write(result["sources"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please try again with a different question.")
