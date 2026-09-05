import os
import json
import urllib.request
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add workspace directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

# Import RAG pipeline modules
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from src.youtube_loader import load_transcript, extract_video_id
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore, load_vectorstore, vectorstore_exists
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain

# --- FastAPI App Initialization ---
app = FastAPI(
    title="YouTube RAG API",
    description="FastAPI Backend for YouTube Video Transcript Question Answering with Hybrid Search and Reranking",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for embeddings model and active RAG chains
EMBEDDINGS = None
RAG_CHAINS_CACHE: Dict[str, Any] = {}

def get_global_embeddings():
    global EMBEDDINGS
    if EMBEDDINGS is None:
        EMBEDDINGS = get_embeddings()
    return EMBEDDINGS

# --- Pydantic Request & Response Schemas ---
class ProcessVideoRequest(BaseModel):
    url_or_id: str = Field(..., example="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

class VideoMetadata(BaseModel):
    title: str
    author: str
    thumbnail_url: str

class ProcessVideoResponse(BaseModel):
    video_id: str
    status: str
    message: str
    is_cached: bool
    metadata: VideoMetadata

class ChatQueryRequest(BaseModel):
    video_id: str = Field(..., example="dQw4w9WgXcQ")
    question: str = Field(..., example="What is this video about?")

class CitationSource(BaseModel):
    start_time_str: str
    end_time_str: str
    text: str
    url: str

class ChatQueryResponse(BaseModel):
    video_id: str
    question: str
    answer: str
    sources: List[CitationSource]

class VideoStatusResponse(BaseModel):
    video_id: str
    exists: bool
    is_loaded_in_memory: bool

# --- Helper Functions ---
def get_youtube_metadata(video_id: str) -> dict:
    url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return {
                "title": data.get("title", "YouTube Video"),
                "author": data.get("author_name", "Unknown Creator"),
                "thumbnail_url": data.get("thumbnail_url", "")
            }
    except Exception:
        return {
            "title": "YouTube Video",
            "author": "Unknown Creator",
            "thumbnail_url": ""
        }

def format_time(seconds: float) -> str:
    secs = int(seconds)
    mins = secs // 60
    hours = mins // 60
    if hours > 0:
        return f"{hours:02d}:{mins%60:02d}:{secs%60:02d}"
    return f"{mins:02d}:{secs%60:02d}"

def get_or_create_rag_chain_for_video(video_id: str):
    if video_id in RAG_CHAINS_CACHE:
        return RAG_CHAINS_CACHE[video_id]

    embeddings = get_global_embeddings()
    if vectorstore_exists(video_id):
        db = load_vectorstore(embeddings, video_id)
        transcript_list, _ = load_transcript(video_id)
        docs = split_documents(transcript_list, video_id)
        retriever = get_retriever(db, docs)
        chain = create_rag_chain(retriever)
        RAG_CHAINS_CACHE[video_id] = chain
        return chain

    # If index doesn't exist, process from scratch
    transcript_list, _ = load_transcript(video_id)
    docs = split_documents(transcript_list, video_id)
    db = create_vectorstore(docs, embeddings, video_id)
    retriever = get_retriever(db, docs)
    chain = create_rag_chain(retriever)
    RAG_CHAINS_CACHE[video_id] = chain
    return chain

# --- API Endpoints ---
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "service": "YouTube RAG FastAPI Backend",
        "version": "1.0.0"
    }

@app.get("/api/video/{video_id}/status", response_model=VideoStatusResponse, tags=["Video"])
def check_video_status(video_id: str):
    exists = vectorstore_exists(video_id)
    is_loaded = video_id in RAG_CHAINS_CACHE
    return VideoStatusResponse(
        video_id=video_id,
        exists=exists,
        is_loaded_in_memory=is_loaded
    )

@app.get("/api/video/{video_id}/metadata", response_model=VideoMetadata, tags=["Video"])
def fetch_video_metadata(video_id: str):
    meta = get_youtube_metadata(video_id)
    return VideoMetadata(**meta)

@app.post("/api/process", response_model=ProcessVideoResponse, tags=["Video"])
def process_video(request: ProcessVideoRequest):
    try:
        video_id = extract_video_id(request.url_or_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube URL or Video ID: {str(e)}")

    metadata = get_youtube_metadata(video_id)
    is_cached = vectorstore_exists(video_id)

    try:
        get_or_create_rag_chain_for_video(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=422, detail="Subtitles/transcripts disabled by video creator.")
    except NoTranscriptFound:
        raise HTTPException(status_code=422, detail="No supported transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    return ProcessVideoResponse(
        video_id=video_id,
        status="success",
        message="Vector database and RAG chain initialized successfully." if not is_cached else "Loaded existing index from disk.",
        is_cached=is_cached,
        metadata=VideoMetadata(**metadata)
    )

@app.post("/api/chat", response_model=ChatQueryResponse, tags=["RAG Chat"])
def chat_with_video(request: ChatQueryRequest):
    video_id = request.video_id
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        chain = get_or_create_rag_chain_for_video(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=422, detail="Transcripts disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=422, detail="No transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load RAG chain for video {video_id}: {str(e)}")

    try:
        response = chain.invoke(request.question)
        answer = response.get("answer", "")
        retrieved_docs = response.get("context", [])

        sources = []
        for doc in retrieved_docs:
            start_time = doc.metadata.get("start_time", 0.0)
            end_time = doc.metadata.get("end_time", 0.0)
            doc_vid = doc.metadata.get("video_id", video_id)

            sources.append(CitationSource(
                start_time_str=format_time(start_time),
                end_time_str=format_time(end_time),
                text=doc.page_content,
                url=f"https://youtu.be/{doc_vid}?t={int(start_time)}"
            ))

        return ChatQueryResponse(
            video_id=video_id,
            question=request.question,
            answer=answer,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
