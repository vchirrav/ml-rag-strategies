"""Hierarchical RAG: Use summaries for coarse retrieval, then drill into details."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def build_hierarchy(sections: dict[str, list[str]], llm: ChatOpenAI):
    """Build a two-level index: summaries (L1) and detailed chunks (L2)."""
    summary_docs = []
    detail_docs = []

    for title, chunks in sections.items():
        # L2: detailed chunks with section metadata
        for chunk in chunks:
            detail_docs.append(
                Document(page_content=chunk, metadata={"section": title})
            )

        # L1: summarize the section
        full_section = "\n".join(chunks)
        summary = llm.invoke(
            f"Summarize in 1-2 sentences:\n{full_section}"
        ).content
        summary_docs.append(
            Document(page_content=summary, metadata={"section": title})
        )

    return summary_docs, detail_docs


def hierarchical_rag(query: str, sections: dict[str, list[str]]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    summary_docs, detail_docs = build_hierarchy(sections, llm)

    # L1: find relevant sections via summaries
    summary_store = FAISS.from_documents(summary_docs, embeddings)
    relevant_sections = summary_store.similarity_search(query, k=2)
    section_names = {doc.metadata["section"] for doc in relevant_sections}

    # L2: retrieve detailed chunks from relevant sections only
    filtered_details = [d for d in detail_docs if d.metadata["section"] in section_names]
    if not filtered_details:
        return "No relevant sections found."

    detail_store = FAISS.from_documents(filtered_details, embeddings)
    top_details = detail_store.similarity_search(query, k=3)

    context = "\n".join(doc.page_content for doc in top_details)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    sections = {
        "Retrieval": [
            "Retrieval uses vector similarity to find relevant documents.",
            "BM25 and dense retrieval are common approaches.",
        ],
        "Generation": [
            "The LLM generates answers conditioned on retrieved context.",
            "Prompt engineering helps control generation quality.",
        ],
        "Evaluation": [
            "RAG systems are evaluated on faithfulness and relevance.",
            "Metrics include RAGAS, answer correctness, and context precision.",
        ],
    }
    print(hierarchical_rag("How is RAG generation evaluated?", sections))
