from rag_engine.rag_chain import qa_chain


def get_recommendation(question):

    query = (
        f"Untuk mahasiswa Ilmu Komputer, "
        f"{question}"
    )

    result = qa_chain.invoke(query)

    return result["result"]