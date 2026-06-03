from rag_engine.vector_store import get_vector_store

vectorstore = get_vector_store()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)