import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever, Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional

from config.settings import DATA_SOURCE_PATH


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"\w+", text.lower())


def load_csv_documents():
    """Load raw CSV data and split into document chunks."""
    if not os.path.exists(DATA_SOURCE_PATH):
        raise FileNotFoundError(
            f"File tidak ditemukan: {DATA_SOURCE_PATH}"
        )

    loader = CSVLoader(file_path=DATA_SOURCE_PATH, encoding="utf-8")
    try:
        docs = loader.load()
    except UnicodeDecodeError:
        loader = CSVLoader(file_path=DATA_SOURCE_PATH, encoding="cp1252")
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


class BM25Retriever(BaseRetriever):
    """Simple BM25 retriever built from the same CSV corpus."""

    documents: List[Document]
    corpus: List[List[str]]
    bm25: BM25Okapi
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: Optional[List[Document]] = None, k: int = 4, **kwargs):
        documents = documents or load_csv_documents()
        corpus = [tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(corpus)
        super().__init__(documents=documents, corpus=corpus, bm25=bm25, k=k, **kwargs)

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][: self.k]
        return [self.documents[i] for i in top_indices]

    def retrieve(self, query: str, k: int = None):
        if k is not None:
            self.k = k
        return self._get_relevant_documents(query)


def build_bm25_retriever():
    return BM25Retriever()
