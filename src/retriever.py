try:
    from langchain_community.retrievers import BM25Retriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.retrievers import BM25Retriever
    except (ImportError, ModuleNotFoundError):
        BM25Retriever = None

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
    except (ImportError, ModuleNotFoundError):
        EnsembleRetriever = None

from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder


def get_retriever(vectorstore, chunks):
    """
    Builds a hybrid retriever combining dense (FAISS) and sparse (BM25) search,
    followed by CrossEncoder reranking.

    Returns a LangChain-compatible Runnable that accepts a query string
    and returns a list of top-5 reranked Document objects.
    """
    # Dense retriever (semantic similarity via FAISS)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    # Hybrid search if both EnsembleRetriever and BM25Retriever are available
    if EnsembleRetriever is not None and BM25Retriever is not None:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
    else:
        # Fallback to dense vector retriever if Ensemble/BM25 module is missing
        hybrid_retriever = vector_retriever

    # Lazy-load CrossEncoder once per call to get_retriever
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _retrieve_and_rerank(query: str):
        docs = hybrid_retriever.invoke(query)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:5]]

    # Wrap as a LangChain Runnable so it's compatible with LCEL chains
    return RunnableLambda(_retrieve_and_rerank)