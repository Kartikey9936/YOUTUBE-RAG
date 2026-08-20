import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src import config

def get_vectorstore_path(video_id: str) -> Path:
    """Returns the subfolder path where FAISS files are saved for a specific video."""
    return Path(config.VECTORSTORE_DIR) / video_id

def vectorstore_exists(video_id: str) -> bool:
    """Checks if the FAISS vectorstore files exist for a specific video."""
    path = get_vectorstore_path(video_id)
    return (path / "index.faiss").exists()

def create_vectorstore(documents: list, embeddings, video_id: str) -> FAISS:
    """
    Creates a new FAISS vector database from document chunks,
    saves it to disk under the video_id subfolder, and returns the instance.
    """
    vectorstore = FAISS.from_documents(documents, embeddings)
    save_path = get_vectorstore_path(video_id)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    return vectorstore

def load_vectorstore(embeddings, video_id: str) -> FAISS:
    """
    Loads an existing FAISS vector database from disk for a specific video.
    """
    load_path = get_vectorstore_path(video_id)
    if not vectorstore_exists(video_id):
        raise FileNotFoundError(f"No vectorstore found for video ID: {video_id} at {load_path}")
    # We must allow dangerous deserialization because we control the database locally
    return FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
