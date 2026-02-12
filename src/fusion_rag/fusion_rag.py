"""Fusion RAG (RAG-Fusion): Combine multiple retrieval results using Reciprocal Rank Fusion."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """Fuse multiple ranked lists using RRF scoring."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in sorted_docs]


def fusion_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    # Generate query variants
    variant_prompt = (
        "Generate 3 alternative search queries for:\n"
        f"{query}\n\nReturn one per line, no numbering."
    )
    variants = llm.invoke(variant_prompt).content.strip().split("\n")
    all_queries = [query] + [v.strip() for v in variants if v.strip()]

    # Retrieve for each query
    ranked_lists = [store.similarity_search(q, k=5) for q in all_queries]

    # Fuse results with RRF
    fused = reciprocal_rank_fusion(ranked_lists)
    top_docs = fused[:3]

    context = "\n".join(doc.page_content for doc in top_docs)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "RAG-Fusion combines results from multiple query variants.",
        "Reciprocal Rank Fusion scores documents across ranked lists.",
        "Diverse queries capture different aspects of user intent.",
        "Ensemble retrieval outperforms single-query approaches.",
        "BM25 and dense retrieval can be fused for hybrid search.",
    ]
    print(fusion_rag("What is RAG-Fusion and how does it work?", corpus))
