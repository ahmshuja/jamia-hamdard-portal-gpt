import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import fitz

# Page config
st.set_page_config(
    page_title="Jamia Hamdard Chatbot",
    page_icon="🎓"
)

st.title("🎓 Jamia Hamdard University Chatbot")
st.markdown("Ask anything about admissions, courses, and fee structure!")

# Load data only once
@st.cache_resource
def load_vectorstore():
    all_docs = []

    # Load text files
    for filename in ["admission.txt", "courses.txt", "fees.txt"]:        
        loader = TextLoader(f"data/{filename}", encoding="utf-8")
        all_docs += loader.load()

    # Load PDFs
    for pdfname in ["Fee_structure.pdf", "Prospectus_2026-27_@_24_March.pdf"]:
        pdf = fitz.open(f"data/{pdfname}")
        for page in pdf:
            text = page.get_text()
            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": pdfname}
                ))

    # Split
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

with st.spinner("Loading Jamia Hamdard data..."):
    vectorstore = load_vectorstore()

st.success("✅ Chatbot Ready!")

# Suggested questions
st.markdown("### 💡 Suggested Questions")
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Fee Structure"):
        st.session_state.query = "What is the fee structure?"
    if st.button("📚 Admission Procedure"):
        st.session_state.query = "What is the admission procedure?"
with col2:
    if st.button("📄 Prospectus"):
        st.session_state.query = "Tell me about Jamia Hamdard prospectus"
    if st.button("📝 Documents Required"):
        st.session_state.query = "What documents are required for admission?"

# Search box
st.markdown("### 🔍 Ask Your Question")
query = st.text_input("Type your question here:", 
                       value=st.session_state.get("query", ""))

if (st.button("Search 🔍") or query) and query:
    with st.spinner("Searching..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke(query)
        answer = retrieved_docs[0].page_content if retrieved_docs else "No answer found"

    st.markdown("### 💬 Answer")
    st.info(answer)
    
    if len(retrieved_docs) > 1:
        with st.expander("📄 See more related information"):
            st.write(retrieved_docs[1].page_content)