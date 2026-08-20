# YouTube Transcript RAG — Implementation Guide

## 1. Project Goal

Convert the existing YouTube Transcript RAG notebook into a modular Streamlit application.

The application should:

1. Accept a YouTube video URL.
2. Fetch the video's transcript.
3. Split the transcript into meaningful chunks.
4. Generate embeddings for the chunks.
5. Store the embeddings in a vector database.
6. Retrieve relevant chunks for a user's question.
7. Send the retrieved context to an LLM.
8. Display the answer in a chat-style Streamlit interface.

## 2. Recommended Project Structure

```text
youtube-rag/
├── app.py
├── src/
│   ├── __init__.py
│   ├── youtube_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── config.py
├── data/
│   └── vectorstore/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Module Responsibilities

### `app.py`
Handles the Streamlit UI:
- YouTube URL input
- Process Video button
- Chat interface
- Status/errors
- `st.session_state` for the RAG chain

Keep the RAG implementation out of this file.

### `src/youtube_loader.py`
Fetches the transcript.

```python
def load_transcript(youtube_url):
    # Fetch transcript and return documents
    return documents
```

### `src/text_splitter.py`
Splits transcript documents into chunks.

```python
def split_documents(documents):
    return chunks
```

Keep chunk size and overlap configurable.

### `src/embeddings.py`
Creates the embedding model.

```python
def get_embeddings():
    return embeddings
```

### `src/vectorstore.py`
Creates and loads the vector database.

```python
def create_vectorstore(chunks, embeddings):
    return vectorstore

def load_vectorstore(embeddings):
    return vectorstore
```

FAISS or Chroma are suitable local options.

### `src/retriever.py`
Creates the retriever.

```python
def get_retriever(vectorstore):
    return vectorstore.as_retriever()
```

### `src/rag_chain.py`
Connects retriever, prompt, and LLM.

```python
def create_rag_chain(retriever):
    return rag_chain
```

### `src/config.py`
Centralizes model names, chunk settings, retriever settings, and vector-store paths.

Do not store API keys here; use `.env`.

## 4. Data Flow

### Video processing

```text
YouTube URL
    ↓
Transcript
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector DB
    ↓
Retriever
    ↓
RAG Chain
```

### Question answering

```text
Question
    ↓
Retriever
    ↓
Relevant transcript chunks
    ↓
Prompt + context + question
    ↓
LLM
    ↓
Answer
```

## 5. Streamlit Flow

Example structure for `app.py`:

```python
import streamlit as st

from src.youtube_loader import load_transcript
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

st.title("YouTube Transcript RAG")

youtube_url = st.text_input("Enter YouTube URL")

if st.button("Process Video"):
    documents = load_transcript(youtube_url)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    st.session_state["rag_chain"] = create_rag_chain(retriever)
    st.success("Video processed!")

if "rag_chain" in st.session_state:
    question = st.chat_input("Ask something about the video")

    if question:
        answer = st.session_state["rag_chain"].invoke(question)
        st.write(answer)
```

## 6. Session State

Streamlit reruns the script during interaction. Do not rebuild the vector database for every question.

Store the RAG chain with:

```python
st.session_state["rag_chain"] = rag_chain
```

For expensive resources, consider Streamlit caching.

## 7. Environment Variables

Create `.env`:

```text
OPENAI_API_KEY=your_api_key
```

Use the appropriate key for your LLM provider.

Never commit `.env`.

`.gitignore` should contain:

```text
.env
venv/
__pycache__/
*.pyc
```

## 8. Requirements

Start with:

```text
streamlit
python-dotenv
langchain
langchain-community
langchain-core
youtube-transcript-api
faiss-cpu
```

Add the provider-specific LLM and embedding packages used by the notebook.

Once the project is stable, pin versions.

## 9. Error Handling

Handle:
- Invalid YouTube URLs
- Missing/unavailable transcripts
- Missing API keys
- Empty questions
- Embedding failures
- Vector database failures
- LLM/API failures

Show clear Streamlit messages instead of crashing.

## 10. Metadata

Preserve useful metadata when possible:

```text
video_id
video_url
video_title
source
chunk_id
start_time
end_time
```

This enables citations, timestamps, source previews, and multi-video retrieval later.

## 11. Development Phases

### Phase 1 — Basic RAG
```text
YouTube URL
→ Transcript
→ Chunks
→ Embeddings
→ Vector DB
→ Retriever
→ LLM
```

### Phase 2 — Streamlit UI
- URL input
- Process button
- Chat history
- Progress/status messages
- Clear chat

### Phase 3 — Retrieval improvements
- Tune chunk size
- Tune overlap
- Tune `k`
- Similarity thresholds
- Reranking

### Phase 4 — Citations
Return the answer with relevant transcript chunks and timestamps.

### Phase 5 — Multiple videos
Allow multiple videos and cross-video questions.

## 12. Recommended Implementation Order

```text
1. Move transcript loading from notebook
2. Move chunking
3. Move embeddings
4. Move vector database
5. Move retriever
6. Move RAG chain
7. Test the modular pipeline
8. Build Streamlit UI
9. Add session state
10. Add error handling
11. Add citations/timestamps
12. Deploy
```

Do not rewrite working RAG logic unnecessarily. First separate the notebook into reusable functions, test those functions, and then connect them to Streamlit.

## 13. Run the Application

From the project root:

```powershell
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 14. Final Architecture

```text
                 Streamlit
                    │
                    ▼
                  app.py
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Video Processing       User Question
          │                   │
          ▼                   ▼
youtube_loader.py       retriever.py
          │                   │
          ▼                   ▼
text_splitter.py        vectorstore.py
          │
          ▼
   embeddings.py
          │
          └─────────┬─────────┘
                    ▼
               rag_chain.py
                    │
                    ▼
                   LLM
                    │
                    ▼
                 Answer
```

## 15. Success Criteria

The project is complete when:

- A user can enter a YouTube URL.
- The transcript is fetched.
- The transcript is chunked.
- Embeddings are generated.
- A vector database is created.
- Questions can be asked through Streamlit.
- Relevant transcript chunks are retrieved.
- The LLM answers using retrieved context.
- The vector database is not rebuilt for every question.
- Errors are clearly displayed.
