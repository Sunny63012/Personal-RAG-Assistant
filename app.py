import streamlit as st
from src.loaders import load_documents
from src.database import create_vector_store
from src.chatbot import generate_response
from langchain_core.messages import HumanMessage,AIMessage
# UPLOAD_DIR="data/uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
st.set_page_config(
    page_title="Personal RAG Assistant",
    layout="wide")
st.title("📚 Personal RAG Assistant")
# --- session state defaults ---
for key, default in [
    ("messages", []),
    ("chat_history", []),
    ("vector_store", None),
    ("bm25", None),
    ("all_chunks", None),
    ("tables",{})
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.subheader("Session")
    st.caption("Documents you upload only exist for this session — nothing is saved between visits.")
    if st.button("🔄 Start new session (clear everything)"):
        for key in ["messages", "chat_history", "vector_store", "bm25", "all_chunks","tables"]:
            st.session_state[key] = [] if key in ("messages", "chat_history") else ({} if key == "tables" else None)
        st.rerun()
uploaded_files=st.file_uploader("Upload your documents here", type=["pdf", "docx", "txt","xlsx","xls","csv"], key="file_uploader",accept_multiple_files=True)
# if uploaded_files:
#     saved_files = []
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())#This method returns the raw binary data (bytes) of the uploaded file.
#         saved_files.append(file_path)
#     st.success(f"Files uploaded successfully!")
#     if st.button("Process Files"):
#         with st.spinner("Loading documents..."):
#             documents=load_documents(saved_files)
#         st.success("Documents loaded successfully!")
#         #t.write(documents[:2])  # Display the first two documents for preview
#         # if st.button("Create Vector Store"):
#         with st.spinner("Creating vector store..."):
#              vector_store, bm25,all_chunks=create_vector_store(documents)
#         st.session_state.vector_store=vector_store
#         st.session_state.bm25=bm25
#         st.session_state.all_chunks=all_chunks
#         st.success("Vector store created successfully!")
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history=[]
if uploaded_files and st.button("Process Files"):
    try:
        with st.spinner("Reading documents..."):
            documents,tables= load_documents(uploaded_files)
        if not documents:
            st.warning("No readable content found in the uploaded files.")
        else:
            st.success(f"Read {len(documents)} document section(s).")
            with st.spinner("Creating vector store..."):
                vector_store, bm25, all_chunks = create_vector_store(documents)
            st.session_state.vector_store = vector_store
            st.session_state.bm25 = bm25
            st.session_state.all_chunks = all_chunks
            st.session_state.tables=tables
            st.success("Ready — you can start asking questions below.")
    except Exception as e:
        st.error(f"Something went wrong while processing your files: {e}")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if st.session_state.vector_store is None:
    st.info("Upload and process documents to start chatting.")
    st.stop()
query=st.chat_input("Ask a question about your documents:")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
            st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response,sources=generate_response(query,st.session_state.vector_store,st.session_state.bm25,st.session_state.all_chunks,st.session_state.chat_history[-5:],st.session_state.tables)
        st.markdown(response)
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"- {source}") 
    st.session_state.chat_history.append(
            HumanMessage(content=query)
        )
    st.session_state.chat_history.append(
            AIMessage(content=response)
        )
    st.session_state.messages.append({"role": "assistant", "content": response}) 
            