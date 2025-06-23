import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# Set OpenAI API key for Langchain/OpenAI usage from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_list.append(text)
                sources_list.append(f"{file.name}_page_{i}")
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            text_list.append(text)
            sources_list.append(file.name)
        else:
            st.warning(f"Unsupported file type: {file.type}")
    return text_list, sources_list

st.set_page_config(layout="centered", page_title="Multidoc_QnA")
st.header("Multidoc_QnA")
st.write("---")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt","pdf"])
st.write("---")

if not uploaded_files:
    st.info("Upload files to analyse")
else:
    st.write(f"{len(uploaded_files)} document(s) loaded..")
    
    documents, sources = read_and_textify(uploaded_files)
    
    # Create embeddings (uses OPENAI_API_KEY from environment)
    embeddings = OpenAIEmbeddings()

    # Create vector store with metadata for source tracking
    vstore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    # Setup retriever
    retriever = vstore.as_retriever(search_kwargs={"k": 2})

    # Initialize ChatOpenAI model with streaming enabled
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True)

    # Create Retrieval QA chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

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
            st.error("Oops, the GPT response resulted in an error :( Please try again with a different question.")
