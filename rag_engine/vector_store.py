import os

from langchain_chroma import Chroma

from config.settings import VECTOR_DB_PATH
from rag_engine.embeddings import embeddings
from rag_engine.ingestion import create_vector_db


def get_vector_store():

    if (
        not os.path.exists(VECTOR_DB_PATH)
        or not os.listdir(VECTOR_DB_PATH)
    ):
        print("Membuat Vector Database...")
        return create_vector_db()

    print("Memuat Vector Database...")

    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
