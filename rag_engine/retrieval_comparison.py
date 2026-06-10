import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain.chains import RetrievalQA
from rag_engine.llm import llm
from rag_engine.prompt import PROMPT
from rag_engine.retriever import retriever, bm25_retriever, compare_retrieval


def print_documents(label, documents):
    print(f"\n=== {label} ===")
    for index, doc in enumerate(documents, start=1):
        preview = doc.page_content.replace("\n", " ")[:300]
        print(f"\n[{index}] {preview}")


def build_qa_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )


def compare_answers(query: str):
    dense_chain = build_qa_chain(retriever)
    bm25_chain = build_qa_chain(bm25_retriever)

    dense_answer = dense_chain.run(query)
    bm25_answer = bm25_chain.run(query)

    return {
        "dense_answer": dense_answer,
        "bm25_answer": bm25_answer,
    }


if __name__ == "__main__":
    query = input("Masukkan query untuk dibandingkan: ")

    result = compare_retrieval(query, k=4)
    print_documents("Dense (Cosine) Retrieval", result["dense"])
    print_documents("BM25 Retrieval", result["bm25"])

    print("\n=== Jawaban dari LLM ===")
    answers = compare_answers(query)
    print("\n[Dense (Cosine) Answer]:\n")
    print(answers["dense_answer"].strip())
    print("\n[BM25 Answer]:\n")
    print(answers["bm25_answer"].strip())
