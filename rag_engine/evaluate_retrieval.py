import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from rag_engine.llm import llm
from rag_engine.prompt import PROMPT
from rag_engine.retriever import bm25_retriever, retriever


def parse_price(price_text: str) -> Optional[int]:
    match = re.search(r"Rp\s*([0-9\.]+)", price_text)
    if not match:
        return None
    return int(match.group(1).replace(".", ""))


def precision_at_k(retrieved_docs: List, relevance_fn: Callable[[str], bool], k: int = 3) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if relevance_fn(doc.page_content))
    return relevant_count / k


def average_precision(retrieved_docs: List, relevance_fn: Callable[[str], bool], k: int = 3) -> float:
    relevant_found = 0
    precision_sum = 0.0

    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        if relevance_fn(doc.page_content):
            relevant_found += 1
            precision_sum += relevant_found / rank

    if relevant_found == 0:
        return 0.0

    return precision_sum / relevant_found


def extract_text_from_response(response) -> str:
    if isinstance(response, dict):
        for key in ("content", "output_text", "result", "text"):
            if response.get(key):
                return response[key]
        return str(response)

    for attr in ("content", "output_text", "result", "text"):
        value = getattr(response, attr, None)
        if value:
            return value

    return str(response)


def normalize_text(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"\w+", text.lower())
        if token not in {
            "yang",
            "dengan",
            "untuk",
            "di",
            "dan",
            "atau",
            "ini",
            "itu",
            "adalah",
            "saya",
            "kamu",
            "ke",
            "dari",
            "sebagai",
            "juga",
        }
    ]


def answer_relevance_score(answer: str, query: str) -> float:
    if not answer or answer.startswith("[Jawaban tidak tersedia"):
        return 0.0
    answer_tokens = set(normalize_text(answer))
    query_tokens = set(normalize_text(query))
    if not query_tokens:
        return 0.0
    overlap = answer_tokens & query_tokens
    return min(1.0, len(overlap) / len(query_tokens))


def truncate_answer(answer: str, max_chars: int = 180) -> str:
    stripped = " ".join(answer.strip().split())
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "..."


def answer_from_documents(query: str, documents: List, max_context_chars: int = 3000) -> str:
    context = "\n\n".join(doc.page_content for doc in documents)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    prompt_text = PROMPT.format(context=context, question=query)
    try:
        response = llm.invoke(prompt_text, generation_config={"max_output_tokens": 80})
        return extract_text_from_response(response)
    except Exception as exc:
        return f"[Jawaban tidak tersedia: {exc}]"


def evaluate_query(query: str, relevance_fn: Callable[[str], bool], k: int = 3) -> Dict[str, float]:
    dense_docs = retriever.get_relevant_documents(query)
    bm25_docs = bm25_retriever.retrieve(query, k=k)

    dense_answer = truncate_answer(answer_from_documents(query, dense_docs[:k]))
    bm25_answer = truncate_answer(answer_from_documents(query, bm25_docs[:k]))

    return {
        "query": query,
        "dense_precision@k": precision_at_k(dense_docs, relevance_fn, k=k),
        "bm25_precision@k": precision_at_k(bm25_docs, relevance_fn, k=k),
        "dense_ap": average_precision(dense_docs, relevance_fn, k=k),
        "bm25_ap": average_precision(bm25_docs, relevance_fn, k=k),
        "dense_answer": dense_answer.strip(),
        "bm25_answer": bm25_answer.strip(),
        "dense_answer_relevancy": answer_relevance_score(dense_answer, query),
        "bm25_answer_relevancy": answer_relevance_score(bm25_answer, query),
    }


def macro_average(scores: List[float]) -> float:
    return float(np.mean(scores)) if scores else 0.0


def get_evaluation_queries():
    return [
        {
            "query": "laptop gaming budget 10 juta",
            "relevance_fn": lambda text: any(
                term in text.lower()
                for term in ["rtx", "gtx", "gaming", "3060", "3050"]
            ) and parse_price(text) is not None and parse_price(text) <= 10_000_000,
        },
        {
            "query": "laptop desain grafis",
            "relevance_fn": lambda text: any(
                term in text.lower()
                for term in ["rtx", "gtx", "nvidia", "amd", "iris xe", "vega"]
            ),
        },
        {
            "query": "laptop untuk mahasiswa mobile dan ringan",
            "relevance_fn": lambda text: any(
                term in text.lower()
                for term in ["14'", "14 inch", "14''", "15'", "15 inch"]
            ) and parse_price(text) is not None and parse_price(text) <= 10_000_000,
        },
        {
            "query": "laptop dengan RAM 16GB",
            "relevance_fn": lambda text: "16gb" in text.lower(),
        },
    ]


def evaluate_all(k: int = 3):
    query_results = []
    dense_precisions = []
    bm25_precisions = []
    dense_aps = []
    bm25_aps = []
    dense_answer_relevancies = []
    bm25_answer_relevancies = []

    for item in get_evaluation_queries():
        result = evaluate_query(item["query"], item["relevance_fn"], k=k)
        query_results.append(result)
        dense_precisions.append(result["dense_precision@k"])
        bm25_precisions.append(result["bm25_precision@k"])
        dense_aps.append(result["dense_ap"])
        bm25_aps.append(result["bm25_ap"])
        dense_answer_relevancies.append(result["dense_answer_relevancy"])
        bm25_answer_relevancies.append(result["bm25_answer_relevancy"])

    report = {
        "dense_macro_precision@k": macro_average(dense_precisions),
        "bm25_macro_precision@k": macro_average(bm25_precisions),
        "dense_macro_ap": macro_average(dense_aps),
        "bm25_macro_ap": macro_average(bm25_aps),
        "dense_macro_answer_relevancy": macro_average(dense_answer_relevancies),
        "bm25_macro_answer_relevancy": macro_average(bm25_answer_relevancies),
        "query_results": query_results,
    }
    return report


def print_table(headers: List[str], rows: List[List[str]]):
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]
    header_row = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
    separator = "-+-".join("-" * width for width in col_widths)
    print(header_row)
    print(separator)
    for row in rows:
        print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
    print()


def print_report(report: Dict[str, any], k: int = 3):
    print("=== METRIK EVALUASI RETRIEVER ===")
    print("Precision@3: Proporsi dokumen relevan di antara 3 dokumen teratas yang diambil setiap metode.")
    print()
    print_table(
        ["Metode", f"Precision@{k}"],
        [
            ["Cosine Similarity", f"{report['dense_macro_precision@k']:.3f}"],
            ["BM25", f"{report['bm25_macro_precision@k']:.3f}"],
        ],
    )

    print("Macro Average Precision: Rata-rata Precision pada setiap query untuk mengevaluasi konsistensi retriever.")
    print()
    print_table(
        ["Metode", "Macro Average Precision"],
        [
            ["Cosine Similarity", f"{report['dense_macro_ap']:.3f}"],
            ["BM25", f"{report['bm25_macro_ap']:.3f}"],
        ],
    )

    print("=== METRIK EVALUASI GENERATOR ===")
    print("Answer Relevancy: Skor otomatis berdasarkan kecocokan jawaban dengan pertanyaan.")
    print()
    print_table(
        ["Metode", "Macro Answer Relevancy"],
        [
            ["Cosine Similarity", f"{report['dense_macro_answer_relevancy']:.3f}"],
            ["BM25", f"{report['bm25_macro_answer_relevancy']:.3f}"],
        ],
    )
    answer_rows = []
    for result in report["query_results"]:
        answer_rows.append([
            result["query"],
            f"{result['dense_answer_relevancy']:.3f}",
            f"{result['bm25_answer_relevancy']:.3f}",
        ])
    print_table(
        ["Query", "Cosine Relevancy", "BM25 Relevancy"],
        answer_rows,
    )

    print("Detail jawaban singkat per query:")
    for result in report["query_results"]:
        print(f"Query: {result['query']}")
        print("  Cosine Similarity Answer:")
        print(f"    {result['dense_answer']}")
        print("  BM25 Answer:")
        print(f"    {result['bm25_answer']}\n")

    print("=== EVALUASI TAMBAHAN ===")
    print("Uji Signifikansi Statistik: Mengukur apakah perbedaan kinerja Precision@3 antara kedua metode signifikan.")
    if report.get("p_value") is not None:
        print(f"  Paired t-test p-value: {report['p_value']:.6f}")

    else:
        print("  p-value tidak tersedia. Pastikan scipy terpasang untuk menampilkan uji signifikansi statistik.")
    print()

def main():
    report = evaluate_all(k=3)
    try:
        from scipy.stats import ttest_rel

        dense_scores = [r["dense_precision@k"] for r in report["query_results"]]
        bm25_scores = [r["bm25_precision@k"] for r in report["query_results"]]
        _, p_value = ttest_rel(dense_scores, bm25_scores)
        report["p_value"] = p_value
    except ImportError:
        report["p_value"] = None

    print_report(report, k=3)


if __name__ == "__main__":
    main()
