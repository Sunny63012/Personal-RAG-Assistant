# from FlagEmbedding import FlagReranker
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import CrossEncoder
import numpy as np
import streamlit as st
from src.database import tokenize
@st.cache_resource
def load_reranker():   
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ms-marco-MiniLM-L-6-v2")
    if not os.path.isdir(MODEL_DIR):
        st.error(f"Reranker model not found at {MODEL_DIR}. Make sure it's committed to the repo.")
        st.stop()
    return CrossEncoder(MODEL_DIR)
reranker=load_reranker()
#we can implement rrf with one dictionary and can combine vector_results and bm25_results at a time and loop over them.
def rrf(vector_results, bm25_results, k=60):
    sun={}
    docs={}
    for i,doc in enumerate(vector_results):
        key=(
            doc.page_content,
            doc.metadata.get("filename"),
            doc.metadata.get("page_number"),)
        sun[key]=1/(i+1+k)
        docs[key]=doc
    for i,doc in enumerate(bm25_results):
        key=(
            doc.page_content,
            doc.metadata.get("filename"),
            doc.metadata.get("page_number"),)
        # if key not in sun:
        #     sun[key]=1/(1+i+k)
        # else:
        #     sun[key]+=1/(1+i+k)
        sun[key]=sun.get(key,0)+1/(1+i+k)
        docs[key]=doc
    sorted_docs=sorted(sun.items(),key=lambda x:x[1],reverse=True)
    #st.write("soretd docs",sorted_docs[:10])
    return [docs[key] for key,_ in sorted_docs[:10]]
def hybrid_search(query,vector_store, bm25, all_chunks):
    #Dense Search using NVIDIA embeddings
    if not all_chunks or bm25 is None:
        return []
    vector_results = vector_store.similarity_search(query, k=10)
    #st.write("vector_results",vector_results)
    
    query_tokens=tokenize(query)  
    scores=bm25.get_scores(query_tokens)
    top_indices=np.argsort(scores)[::-1][:10]
    bm25_results=[all_chunks[i] for i in top_indices]
    
    #st.write("bm25 results",bm25_results)
    fused_results=rrf(vector_results,bm25_results)
    if not fused_results:
        return []
    pairs=[(query,doc.page_content) for doc in fused_results]
    scores=reranker.predict(pairs)
    ranked_results=sorted(zip(fused_results,scores),key=lambda x:x[1],reverse=True)
    #st.write("reranked results",ranked_results[:5])
    return [doc for doc,score in ranked_results[:5]]
    