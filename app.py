import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import tempfile

# HuggingFace API Token (for hosted models)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="Product FAQ Chatbot", layout="wide")

st.title("🤖 Product FAQ Chatbot")
st.write("Upload your product documents and ask any question!")

uploaded_file = st.file_uploader("Upload a document (.txt)", type=["txt"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split document
    loader = TextLoader(tmp_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    # Set up retriever and QA chain
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Change to mistral/llama if using local LLMs
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.success("Document processed successfully! You can now ask questions.")

    # Input from user
    user_query = st.text_input("Ask a question about the product:")

    if user_query:
        answer = qa_chain.run(user_query)
        st.markdown("### 💬 Answer")
        st.write(answer)
