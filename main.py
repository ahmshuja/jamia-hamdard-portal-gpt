# Import libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from transformers import pipeline
import fitz
from langchain_core.documents import Document

# STEP 1: Load data
print("Loading data...")
all_docs = []

for filename in ["admission.txt", "courses.txt", "fees.txt"]:
    loader = TextLoader(f"data/{filename}", encoding="utf-8")
    all_docs += loader.load()
    print(f"✅ {filename} loaded")

pdf = fitz.open("data/Fee_structure.pdf")
for i in range(len(pdf)):
    text = pdf[i].get_text()
    if text.strip():
        all_docs.append(Document(page_content=text, metadata={"source": "Fee_structure.pdf"}))
print("✅ Fee_structure.pdf loaded")

# STEP 2: Split
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(all_docs)

# STEP 3: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 4: FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# STEP 5: No LLM needed - retrieval only

# STEP 6: Ask 4 queries
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

queries = [
    "What is the fee structure?",
    "Tell me about Jamia Hamdard prospectus?",
    "What is the admission procedure?",
    "What documents are required for admission?"
]
print("\n🎓 Jamia Hamdard Chatbot")
print("=" * 50)

for query in queries:
    retrieved_docs = retriever.invoke(query)
    answer = retrieved_docs[0].page_content if retrieved_docs else "No answer found"
    print(f"\n❓ {query}")
    print(f"💬 {answer}")
    print("-" * 50)
