# 🎥 YouTube RAG Chatbot

An interactive RAG (Retrieval-Augmented Generation) application built with **Streamlit**, **LangChain**, **FAISS**, and **Groq LLM**. Chat with any YouTube video transcript and get precise, timestamped answer citations.

---

## 🚀 Features

- 📜 **Automatic Transcript Fetching** — Downloads captions from YouTube with language fallback (English → Hindi → any available)
- 🧠 **Local Embeddings** — Powered by `sentence-transformers/all-MiniLM-L6-v2` (runs fully offline)
- ⚡ **Fast Generation** — Backed by Groq LLM (`openai/gpt-oss-20b`)
- ⏱️ **Timestamp Citations** — Click-to-watch links that jump to the exact moment in the video
- 🔍 **Hybrid Search** *(v2)* — Combines dense FAISS vector search with sparse BM25 keyword search via `EnsembleRetriever`
- 🏆 **Cross-Encoder Reranking** *(v2)* — Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the top-20 retrieved chunks and return only the best 5

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (LCEL) |
| Dense Retrieval | FAISS + HuggingFace Embeddings |
| Sparse Retrieval | BM25 (`langchain-community`) |
| Hybrid Fusion | `EnsembleRetriever` (RRF, weights 0.7 / 0.3) |
| Reranker | `CrossEncoder` — ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Vector Store | FAISS (persisted to disk per video) |

---

## ⚙️ How It Works

```
YouTube URL
    │
    ▼
1. Extract transcript (youtube-transcript-api)
    │
    ▼
2. Chunk + timestamp-map (RecursiveCharacterTextSplitter)
    │
    ▼
3. Embed + store in FAISS (cached to disk per video ID)
    │
    ▼
4. On query → Hybrid Retrieval (FAISS 70% + BM25 30%) → top-20 candidates
    │
    ▼
5. CrossEncoder reranker → top-5 most relevant chunks
    │
    ▼
6. Groq LLM generates answer with timestamp citations
```

---

## 📦 Installation

```bash
git clone <repo-url>
cd YOUTUBE-RAG
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
YOUTUBE-RAG/
├── app.py                  # Streamlit UI + pipeline orchestration
├── src/
│   ├── config.py           # Paths, model names, chunk settings
│   ├── youtube_loader.py   # Transcript fetching + video ID extraction
│   ├── text_splitter.py    # Chunking + timestamp metadata tagging
│   ├── embeddings.py       # HuggingFace embedding model loader
│   ├── vectorstore.py      # FAISS create / load / persist
│   ├── retriever.py        # Hybrid retriever + CrossEncoder reranker
│   └── rag_chain.py        # LCEL RAG chain (prompt + LLM + parser)
├── data/
│   └── vectorstore/        # Persisted FAISS indexes (per video ID)
├── requirements.txt
└── .env
```

---

## 🔄 Changelog

### v2 — Hybrid Search & Reranking
- **Added**: `EnsembleRetriever` combining FAISS (dense) + BM25 (sparse) retrieval
- **Added**: `CrossEncoder` reranker (`ms-marco-MiniLM-L-6-v2`) over top-20 candidates → returns top-5
- **Fixed**: `get_retriever()` now returns a LangChain-compatible `RunnableLambda` so the reranker is fully wired into the LCEL chain
- **Fixed**: `CrossEncoder` now loads lazily inside `get_retriever()` instead of at module import time
- **Fixed**: Both `get_retriever()` call sites in `app.py` updated to pass `chunks` (required by BM25)
- **Fixed**: Pre-existing index path now re-fetches transcript chunks for BM25 (FAISS is disk-cached; BM25 is in-memory only)

### v1 — Initial Release
- Basic RAG pipeline with FAISS vector search
- Streamlit chat UI with timestamp citation cards
- Groq LLM integration

---

## 📄 License

MIT
