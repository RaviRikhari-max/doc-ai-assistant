import os
import streamlit as st
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
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
    return vectorstore

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_button and uploaded_files:
    with st.spinner("Processing and indexing documents..."):
        st.session_state.vectorstore = process_documents(uploaded_files)
        st.success("Documents processed successfully! You can now start asking questions.")

if st.session_state.vectorstore:
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, say that you don't know.

        Context:
        {context}

        Question: {query}
        Answer:"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            st.write("### Answer:")
            st.write(response.content)
else:
    st.info("Please upload files and click 'Process Documents' in the sidebar to begin.")
