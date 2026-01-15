from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from prompt import PROMPT

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------------------ CONSTANTS ------------------
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
COLLECTION_NAME = "real_estate"

# ------------------ GLOBALS ------------------
llm = None
vector_store = None


# ------------------ INITIALIZATION ------------------
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ------------------ URL PROCESSING ------------------
def process_urls(urls):
    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()

    yield "Loading data from URLs..."
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    # 🚨 BLOCKED PAGE VALIDATION (IMPORTANT FIX)
    blocked_keywords = ["Access Denied", "403", "Forbidden"]
    valid_docs = []

    for doc in documents:
        if any(word in doc.page_content for word in blocked_keywords):
            yield "⚠️ Skipping blocked webpage"
        else:
            valid_docs.append(doc)

    if not valid_docs:
        yield "❌ No valid content found. All pages were blocked."
        return

    yield "Splitting text into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    docs = splitter.split_documents(valid_docs)

    yield "Adding documents to vector database..."
    ids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=ids)

    yield "Vector database updated successfully ✅"


# ------------------ QA GENERATION (RAG) ------------------
def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {
            "summaries": retriever,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    return answer, []


# ------------------ LOCAL TEST ------------------
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for msg in process_urls(urls):
        print(msg)

    answer, _ = generate_answer(
        "What was the 30 year fixed mortgage rate and the date?"
    )

    print("Answer:", answer)
