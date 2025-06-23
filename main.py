import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        if file.type == "application/pdf":
            pdfReader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdfReader.pages):
                text = page.extract_text()
                text_list.append(text)
                sources_list.append(f"{file.name}_page_{i}")
        else:
            # For txt files, just read content
            text = file.read().decode("utf-8")
            text_list.append(text)
            sources_list.append(file.name)
    return text_list, sources_list

st.set_page_config(layout="centered", page_title="Multidoc_QnA")
st.header("Multidoc_QnA")
st.write("---")

uploaded_files = st.file_uploader(
    "Upload documents", accept_multiple_files=True, type=["txt", "pdf"]
)
st.write("---")

if not uploaded_files:
    st.info("Upload files to analyze")
else:
    st.write(f"{len(uploaded_files)} document(s) loaded...")

    documents, sources = read_and_textify(uploaded_files)

    # Initialize embeddings without model param (uses default text-embedding-ada-002)
    embeddings = OpenAIEmbeddings()

    # Create vector store with metadata
    vstore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    retriever = vstore.as_retriever(search_kwargs={"k": 2})

    # Initialize LLM - specify your model here if you want
    llm = OpenAI(model_name="gpt-3.5-turbo", streaming=True)

    # Setup retrieval QA chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    st.header("Ask your data")
    user_question = st.text_area("Enter your question here")

    if st.button("Get Response"):
        if not user_question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain({"question": user_question}, return_only_outputs=True)
                    st.subheader("Response:")
                    st.write(response["answer"])
                    st.subheader("Source(s):")
                    st.write(response["sources"])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please try again with a different question.")
