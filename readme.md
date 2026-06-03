# Chatbot Rekomendasi Laptop Kendari

Aplikasi RAG (Retrieval‑Augmented Generation) untuk merekomendasikan laptop bagi mahasiswa Kendari. UI utama berbasis Streamlit.

Ringkasan cepat
- UI Streamlit: [app.py](app.py)  
- Styling: [assets/style.css](assets/style.css)  
- Data produk: [data/laptop_kendari.csv](data/laptop_kendari.csv)  
- Vector DB persist: [chroma_db/](chroma_db/)  
- Env: [.env](.env)  
- Dependensi: [requairments.txt](requairments.txt)

Arsitektur & titik integrasi penting
- Service publik: [`services.recommendation_service.get_recommendation`](services/recommendation_service.py) — [services/recommendation_service.py](services/recommendation_service.py)  
- RAG pipeline:
  - Vector store: [`rag_engine.vector_store.get_vector_store`](rag_engine/vector_store.py) — [rag_engine/vector_store.py](rag_engine/vector_store.py)  
  - Ingest data: [`rag_engine.ingestion.create_vector_db`](rag_engine/ingestion.py) — [rag_engine/ingestion.py](rag_engine/ingestion.py)  
  - Embeddings: [`rag_engine.embeddings.embeddings`](rag_engine/embeddings.py) — [rag_engine/embeddings.py](rag_engine/embeddings.py)  
  - Retriever: [`rag_engine.retriever.retriever`](rag_engine/retriever.py) — [rag_engine/retriever.py](rag_engine/retriever.py)  
  - Prompt template: [`rag_engine.prompt.PROMPT`](rag_engine/prompt.py) — [rag_engine/prompt.py](rag_engine/prompt.py)  
  - LLM wrapper: [`rag_engine.llm.llm`](rag_engine/llm.py) — [rag_engine/llm.py](rag_engine/llm.py)  
  - QA chain: [`rag_engine.rag_chain.qa_chain`](rag_engine/rag_chain.py) — [rag_engine/rag_chain.py](rag_engine/rag_chain.py)

Persiapan & Menjalankan
1. Buat virtualenv dan install dependency:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requairments.txt