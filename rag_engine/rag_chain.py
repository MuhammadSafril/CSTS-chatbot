from langchain.chains import RetrievalQA

from rag_engine.llm import llm
from rag_engine.prompt import PROMPT
from rag_engine.retriever import retriever


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PROMPT
    }
)