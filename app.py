import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="centered")

st.title("📚 AI Document Assistant")
st.markdown("Upload your documents (PDF, TXT, DOCX) and chat with them instantly using AI.")

# Retrieve API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found! Please add it to your Streamlit secrets.")
    st.stop()

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files", type=["pdf", "txt", "docx"], accept_multiple_files=True
    )

    process_button = st.button("Process Documents")

# Function to process uploaded files
@st.cache_resource
def process_documents(files):
    docs = []
    for file in files:
        file_path = os.path.join(".", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())

        os.remove(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

if "vectorstore_retriever" not in st.session_state:
    st.session_state.vectorstore_retriever = None

if process_button and uploaded_files:
    with st.spinner("Processing and indexing documents..."):
        st.session_state.vectorstore_retriever = process_documents(uploaded_files)
        st.success("Documents processed successfully! You can now start asking questions.")

if st.session_state.vectorstore_retriever:
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(st.session_state.vectorstore_retriever, question_answer_chain)

        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})
            st.write("### Answer:")
            st.write(response["answer"])
else:
    st.info("Please upload files and click 'Process Documents' in the sidebar to begin.")
