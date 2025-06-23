import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# App layout
st.set_page_config(page_title="MultiDoc Q&A", layout="centered")
st.title("📄 Multi-Document Q&A")
st.write("Upload one or more PDF or TXT files and ask questions about their content.")

# Upload files
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    raw_text = ""
    sources = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        if filename.endswith(".pdf"):
            pdf = PdfReader(uploaded_file)
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
                    sources.append(f"{filename}_page_{page_num}")
        elif filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            raw_text += text + "\n"
            sources.append(filename)

    if not raw_text.strip():
        st.warning("No readable text found in the uploaded files.")
        st.stop()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(raw_text)

    # Generate metadata
    metadatas = [{"source": s} for s in sources for _ in range(len(docs) // len(sources))]

    try:
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(docs, embeddings, metadatas=metadatas)

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            retriever=retriever,
            return_source_documents=True
        )

        st.success("✅ Documents processed. Ask your question below.")
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Thinking..."):
                result = qa_chain.run(question)
                st.write("### 🤖 Answer:")
                st.write(result)
    except Exception as e:
        st.error(f"Embedding or model error: {e}")
else:
    st.info("Please upload some files to get started.")
