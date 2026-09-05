# 🎥 YouTube RAG Chatbot & FastAPI Backend

An interactive RAG (Retrieval-Augmented Generation) application and REST API built with **FastAPI**, **Streamlit**, **LangChain**, **FAISS**, and **Groq LLM**. Chat with any YouTube video transcript and get precise, timestamped answer citations.

---

## 🚀 Features

- ⚡ **FastAPI Backend** — High-performance RESTful API with CORS, automatic Swagger UI docs (`/docs`), and Pydantic validation
- 📜 **Automatic Transcript Fetching** — Downloads captions from YouTube with language fallback (English → Hindi → any available)
- 🧠 **Local Embeddings** — Powered by `sentence-transformers/all-MiniLM-L6-v2` (runs fully offline)
- ⚡ **Fast Generation** — Backed by Groq LLM (`openai/gpt-oss-20b`)
- ⏱️ **Timestamp Citations** — Click-to-watch links that jump to the exact moment in the video
- 🔍 **Hybrid Search** — Combines dense FAISS vector search with sparse BM25 keyword search via `EnsembleRetriever`
- 🏆 **Cross-Encoder Reranking** — Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved chunks and return the best 5

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API Backend | FastAPI + Uvicorn |
| Web UI | Streamlit |
| Orchestration | LangChain (LCEL) |
| Dense Retrieval | FAISS + HuggingFace Embeddings |
| Sparse Retrieval | BM25 (`langchain-community`) |
| Hybrid Fusion | `EnsembleRetriever` (RRF, weights 0.7 / 0.3) |
| Reranker | `CrossEncoder` — ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Vector Store | FAISS (persisted to disk per video) |

---

## 🚀 How to Run

### 1. Installation

```bash
git clone <repo-url>
cd YOUTUBE-RAG
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Running FastAPI Backend Server

```bash
uvicorn main:app --reload --port 8000
```
- **Interactive Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### 3. Running Streamlit Web UI

```bash
streamlit run app.py
```

---

## 🔌 FastAPI Endpoint Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check endpoint |
| `POST` | `/api/process` | Process a YouTube URL or video ID and initialize vector index |
| `POST` | `/api/chat` | Query the RAG chain for a video and receive timestamp citations |
| `GET` | `/api/video/{video_id}/status` | Check if vectorstore index exists on disk or in memory |
| `GET` | `/api/video/{video_id}/metadata` | Fetch YouTube video title, creator name, and thumbnail |

---

## 📁 Project Structure

```
YOUTUBE-RAG/
├── main.py                 # FastAPI backend server
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

## 📄 License

MIT
