# 📚 Personal RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) app built with Streamlit that lets you upload PDFs, Word docs, Excel/CSV files, and text files, then ask natural-language questions about them — with citations back to the exact source.

Built to be genuinely usable day-to-day: multi-user safe, resilient to flaky network calls, and accurate on both prose questions ("what does the syllabus cover?") and tabular questions ("what's the highest price in this dataset?").

---

## ✨ Features

- **Multi-format ingestion:** PDF, DOCX, TXT, XLSX/XLS, CSV
- **Hybrid retrieval:** dense vector search (NVIDIA embeddings via Chroma) fused with BM25 keyword search via Reciprocal Rank Fusion, then reranked with a cross-encoder for precision
- **Structured query engine:** exact ID lookups, multi-condition filters, and aggregations (max/min/average/sum/count) on tabular data are answered with real pandas computation — not LLM guesswork — so numeric questions are exact, not approximate
- **Conversational memory:** follow-up questions ("what did I ask earlier?") are handled using chat history, separately from document retrieval
- **Per-session isolation:** each user's documents and vector store live only in their own session — nothing is shared or persisted across users or reboots
- **Resilient API calls:** automatic retries with exponential backoff and explicit timeouts on all NVIDIA API calls (embeddings + chat), plus batched embedding requests to avoid oversized payloads
- **Source citations:** every answer lists which file, page, sheet, or row it came from

---

## 🏗️ Architecture

```
User uploads files
        │
        ▼
   loaders.py  ──► parses PDF / DOCX / TXT / XLSX / CSV into text chunks
        │              (Excel/CSV also kept as full DataFrames for structured queries)
        ▼
  database.py  ──► splits text, batches + embeds via NVIDIA, builds an
                    in-memory Chroma vector store + BM25 index (per session)
        │
        ▼
   User asks a question
        │
        ▼
  chatbot.py   ──► tries structured_query.py first:
                       1. Exact ID lookup?      → pandas filter
                       2. Multi-condition filter? → pandas filter
                       3. Aggregation (max/min/avg/sum/count)? → pandas computation
                    If none apply, falls back to RAG:
                       retriever.py → hybrid_search()
                         - vector similarity search (top 10)
                         - BM25 keyword search (top 10)
                         - Reciprocal Rank Fusion
                         - cross-encoder reranking (top 5)
                       → context + chat history → NVIDIA LLM → answer
```

**Why structured queries exist:** embedding-based retrieval is built for *semantic similarity*, not *exact match* or *whole-table aggregation*. Asking an LLM to eyeball "which of these 5 retrieved rows has the highest price" out of a 2,000-row dataset is unreliable — it can only see 5 rows at a time, and it isn't doing real arithmetic. Exact lookups, filters, and aggregations are instead answered by running real pandas queries against the full table, which is accurate by construction.

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | NVIDIA NIM (`meta/llama-3.1-8b-instruct`) via `langchain-nvidia-ai-endpoints` |
| Embeddings | NVIDIA `nv-embedqa-e5-v5` |
| Vector store | ChromaDB (in-memory, per-session) |
| Keyword search | `rank-bm25` |
| Reranking | `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| Structured queries | pandas |
| Resilience | `tenacity` (retries), explicit timeouts |
| Document parsing | PyMuPDF, `python-docx`, pandas/openpyxl |

---

## 📁 Project Structure

```
Personal_RAG_Assistant/
├── app.py                  # Streamlit UI, session state, chat loop
├── src/
│   ├── loaders.py          # File parsing (PDF/DOCX/TXT/XLSX/CSV)
│   ├── database.py         # Chunking, embedding, vector store, BM25 index
│   ├── retriever.py        # Hybrid search (vector + BM25 + RRF + rerank)
│   ├── chatbot.py          # Response generation, structured-query dispatch
│   └── structured_query.py # Exact lookups, filters, aggregations via pandas
├── models/
│   └── ms-marco-MiniLM-L-6-v2/  # Local cross-encoder model (must be committed)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml        # NOT committed — API key lives here
└── README.md
```

---

## 🚀 Getting Started (Local)

1. **Clone and set up a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Add your NVIDIA API key** — get one at [build.nvidia.com](https://build.nvidia.com). Use either:

   - `.streamlit/secrets.toml`:
     ```toml
     NVIDIA_API_KEY = "your-key-here"
     ```
   - or a `.env` file in the project root:
     ```
     NVIDIA_API_KEY="your-key-here"
     ```

   ⚠️ Never commit either file — both are covered by `.gitignore`.

3. **Make sure the reranker model is present** at `models/ms-marco-MiniLM-L-6-v2/` (download once from Hugging Face and commit it, since the app runs fully offline for reranking via `HF_HUB_OFFLINE=1`).

4. **Run it**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (confirm `.env`, `_env`, and `secrets.toml` are **not** in it).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. Go to **Settings → Secrets** and add:
   ```toml
   NVIDIA_API_KEY = "your-key-here"
   ```
4. Deploy. Because the vector store is in-memory and session-scoped, no persistent storage setup is required — and no user's documents are ever visible to another user's session.

---

## 🔒 Privacy & Multi-Tenancy Notes

- Uploaded files are processed entirely in memory — nothing is written to disk.
- The vector store, BM25 index, and chat history exist only in that browser session's `st.session_state` and are discarded when the session ends or "Start new session" is clicked.
- This means the app doesn't retain a persistent shared knowledge base between visits — a deliberate tradeoff for privacy on a shared/public deployment. Every session starts from a clean, empty state.

---

## ⚠️ Known Limitations

- **Large tables (1,000+ rows):** structured queries (lookups, filters, aggregations) scale fine since they run on the full DataFrame, but free-text RAG questions over very large sheets are still limited by the top-5-chunks retrieval window.
- **Scanned/image-only PDFs:** text extraction returns nothing for scanned pages — there's no OCR step. The app warns you per-page rather than failing silently.
- **No persistent knowledge base:** by design, nothing carries over between sessions. A shared, always-available team knowledge base would need a different architecture (persistent vector DB + access control).
- **Structured-query pattern matching is heuristic:** ID/filter/aggregation detection relies on keyword and column-name matching rather than a full NL-to-SQL layer, so unusual phrasing may still fall through to plain RAG.

---

## 🗺️ Possible Next Steps

- OCR fallback for scanned PDFs
- Streaming LLM responses for better perceived latency
- Optional persistent, access-controlled knowledge base for team use
- Broader structured-query coverage (e.g. sorting, top-N, grouped aggregations)
