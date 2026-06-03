import os
import uuid

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config.settings import (
    DATA_SOURCE_PATH,
    VECTOR_DB_PATH
)

from rag_engine.embeddings import embeddings


def create_vector_db():
    """
    Membaca dataset laptop dan menyimpannya ke ChromaDB
    """

    if not os.path.exists(DATA_SOURCE_PATH):
        raise FileNotFoundError(
            f"File tidak ditemukan: {DATA_SOURCE_PATH}"
        )

    print(f"Memproses data: {DATA_SOURCE_PATH}")

    try:
        loader = CSVLoader(
            file_path=DATA_SOURCE_PATH,
            encoding="utf-8"
        )
        docs = loader.load()

    except UnicodeDecodeError:

        print("UTF-8 gagal, mencoba CP1252...")

        loader = CSVLoader(
            file_path=DATA_SOURCE_PATH,
            encoding="cp1252"
        )

        docs = loader.load()

    print(f"Jumlah dokumen: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    splits = splitter.split_documents(docs)

    print(f"Jumlah chunk: {len(splits)}")

    # Generate ID manual
    ids = [str(uuid.uuid4()) for _ in range(len(splits))]

    # Hapus database lama jika ada
    if os.path.exists(VECTOR_DB_PATH):
        print("Vector DB ditemukan, menggunakan database yang ada...")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        ids=ids,
        persist_directory=VECTOR_DB_PATH
    )

    print("✅ Vector Database berhasil dibuat!")

    return vectorstore


if __name__ == "__main__":
    create_vector_db()
