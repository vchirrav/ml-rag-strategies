"""Contextual Retrieval: Prepend document-level context to each chunk before embedding."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def add_context_to_chunks(
    full_document: str, chunks: list[str], llm: ChatOpenAI
) -> list[str]:
    """Use an LLM to generate a short context prefix for each chunk."""
    contextualized = []
    for chunk in chunks:
        prompt = (
            f"Document:\n{full_document[:500]}\n\n"
            f"Chunk:\n{chunk}\n\n"
            f"Write a 1-2 sentence context that situates this chunk within the "
            f"full document. Be concise."
        )
        context = llm.invoke(prompt).content
        contextualized.append(f"{context}\n\n{chunk}")
    return contextualized


def contextual_retrieval_rag(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    full_doc = (
        "RAG systems retrieve external knowledge to augment LLM responses. "
        "They consist of an indexing pipeline, a retrieval step, and a generation step. "
        "Contextual retrieval improves chunk quality by adding document-level context."
    )
    raw_chunks = [
        "They consist of an indexing pipeline, a retrieval step, and a generation step.",
        "Contextual retrieval improves chunk quality by adding document-level context.",
    ]

    enriched_chunks = add_context_to_chunks(full_doc, raw_chunks, llm)
    documents = [Document(page_content=c) for c in enriched_chunks]
    store = FAISS.from_documents(
        documents, OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    )

    results = store.similarity_search(query, k=2)
    context = "\n".join(doc.page_content for doc in results)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    print(contextual_retrieval_rag("What is contextual retrieval?"))
