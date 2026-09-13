from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, ResourceExhausted, InternalServerError
import httpx
import streamlit as st
import re
import time
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
# DB_DIR = "data/vector_store"
# os.makedirs(DB_DIR, exist_ok=True)  # Ensure the directory exists
# #@st.cache_resource

BATCH_SIZE=20
BATCH_DELAY_SECONDS = 10
RETRYABLE = (ServiceUnavailable, DeadlineExceeded, ResourceExhausted, InternalServerError, TimeoutError,httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,)
def get_api_key():
    # st.secrets works on Streamlit Cloud (Settings -> Secrets).
    # Falls back to a local .env for development.
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except FileNotFoundError:
        pass
    return os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = "gemini-embedding-2"
@st.cache_resource
def get_embedding_model():

    key = get_api_key()

    if not key:
        st.error(
            "GOOGLE_API_KEY is not set. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=key
    )
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=3, max=90),
    retry=retry_if_exception_type(RETRYABLE),
    reraise=True,
)
def _add_batch(vector_store, batch):
    vector_store.add_documents(batch)
def tokenize(text):
        return re.findall(r"\b\w+\b",text.lower())
def create_vector_store(documents):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    # Split the documents into chunks
    chunks=[]
    for doc in documents:
        if doc.metadata.get("source_type") in ["excel","csv"]:
            # doc=re.findall(r"\b\w+\b",doc.page_content.lower())
            chunks.append(doc)
        else:
            chunks.extend(text_splitter.split_documents([doc]))
    # Initialize the NVIDIA embeddings model
    embedding_model = get_embedding_model()
    # Create embeddings for the text chunks
    vector_store=Chroma( 
        embedding_function=embedding_model)
    progress = st.progress(0.0, text="Embedding your documents...")
    total_batches=max(1,(len(chunks)+BATCH_SIZE-1)//BATCH_SIZE)
    for i in range(0,len(chunks),BATCH_SIZE):
        batch=chunks[i:i+BATCH_SIZE]
        try:
            _add_batch(vector_store,batch)
        except Exception as e:
            st.error(
                f"Failed to embed a batch of chunks after several  retries ({e}). "
                "Some content may be missing from search results."
            )
            raise e
        progress.progress(min(1.0, (i + BATCH_SIZE) / len(chunks)), text="Embedding your documents...")
        time.sleep(BATCH_DELAY_SECONDS)
    progress.empty()
    data=vector_store.get()
    all_chunks=[]
    for text,metadata in zip(data["documents"],data["metadatas"]):
        all_chunks.append(Document(
            page_content=text,
            metadata=metadata
        ))
    tokenized_texts = [tokenize(doc.page_content) for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_texts) if tokenized_texts else None
    return  vector_store,bm25,all_chunks

    







