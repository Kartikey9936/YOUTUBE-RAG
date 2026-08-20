from langchain_core.vectorstores import VectorStoreRetriever
from src import config

def get_retriever(vectorstore) -> VectorStoreRetriever:
    """
    Returns a retriever configured with similarity search and the default 'k' from config.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )
