from rag_engine.vector_store import get_vector_store
from rag_engine.bm25_retriever import build_bm25_retriever

vectorstore = get_vector_store()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

bm25_retriever = build_bm25_retriever()


def compare_retrieval(query: str, k: int = 4):
    """Bandingkan hasil retrieval antara dense dan BM25."""
    dense_docs = retriever.get_relevant_documents(query)
    bm25_docs = bm25_retriever.get_relevant_documents(query, k=k)
    return {
        "dense": dense_docs,
        "bm25": bm25_docs,
    }
