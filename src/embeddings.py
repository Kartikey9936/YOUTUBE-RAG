from langchain_huggingface import HuggingFaceEmbeddings
from src import config

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model specified in config.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )
