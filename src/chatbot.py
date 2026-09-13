from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from src.retriever import hybrid_search
from src.database import get_api_key
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded, ResourceExhausted, InternalServerError
import httpx
import streamlit as st
from src.structured_query import try_exact_lookup, try_aggregation
parser=StrOutputParser()
RETRYABLE = (ServiceUnavailable, DeadlineExceeded, ResourceExhausted, InternalServerError, TimeoutError,httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,)
CHAT_MODEL = "gemini-3.5-flash"
@st.cache_resource
def get_llm():

    api_key = get_api_key()

    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not set. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=1024,
        timeout=45,
    )
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """ 
You are a Personal RAG Assistant.
The context may contain only a subset of the uploaded documents or spreadsheet rows.

Do NOT assume that information missing from the context does not exist in the original documents.

Do NOT infer that an ID, record, property, or value does not exist simply because it is absent from the retrieved context.

You have TWO sources of information:

1. Retrieved document context.
2. Previous conversation history.

Rules:

- If the user's question is about the uploaded documents,
  answer ONLY using the retrieved context.

- If the user's question refers to the previous conversation
  (for example: "What was my first question?",
  "What did I ask earlier?",
  "Summarize our conversation"),
  answer using the chat history.

- If the answer is not available in either the retrieved
  context or the conversation history, reply:

"I couldn't find that information."

Do not hallucinate.

Retrieved Context:

{context}
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}")])
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(RETRYABLE),
    reraise=True,
)
def _invoke_chain(chain, inputs):
    return chain.invoke(inputs)
def generate_response(query,vector_store,bm25,all_chunks,chat_history,tables=None):
    exact = try_exact_lookup(query, tables)
    if exact:
        return exact, ["computed directly from table"]

    # 2. Aggregation (max/min/avg/sum/count) — deterministic pandas computation
    agg = try_aggregation(query, tables)
    if agg:
        return agg, ["computed directly from table"]
    retrieved_docs=hybrid_search(query,vector_store,bm25,all_chunks)
    if not retrieved_docs:
        return "I could not find any information",[]
    context="\n".join([doc.page_content for doc in retrieved_docs])
#     prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """ 
# You are a Personal RAG Assistant.

# You have TWO sources of information:

# 1. Retrieved document context.
# 2. Previous conversation history.

# Rules:

# - If the user's question is about the uploaded documents,
#   answer ONLY using the retrieved context.

# - If the user's question refers to the previous conversation
#   (for example: "What was my first question?",
#   "What did I ask earlier?",
#   "Summarize our conversation"),
#   answer using the chat history.

# - If the answer is not available in either the retrieved
#   context or the conversation history, reply:

# "I couldn't find that information."

# Do not hallucinate.

# Retrieved Context:

# {context}
# """
#     ),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{query}")
# ])
    chain=prompt | get_llm() | parser
    # start=time.time()
    try:
        response=_invoke_chain(chain,{"context":context,"query":query,"chat_history":chat_history})
    except Exception as e:
        return (
            "Sorry — I couldn't reach the model after a few retries "
            f"({type(e).__name__}). Please try again in a moment.",[],)
    # st.write("Time:",time.time()-start)
    sources=[]
    for doc in retrieved_docs:
        file=doc.metadata.get("filename")
        page=doc.metadata.get("page_number") or doc.metadata.get("row_number") or "-"
        source=doc.metadata.get("source_type")
        source_info=f"{file} | {page}| {source}"
        if source_info not in sources:
            sources.append(source_info)
    return response,sources
