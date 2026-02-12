"""HyDE RAG: Hypothetical Document Embeddings - generate a fake answer, embed it, retrieve."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def generate_hypothetical_document(query: str, llm: ChatOpenAI) -> str:
    """Generate a hypothetical answer to use as a retrieval query."""
    prompt = (
        "Write a short paragraph that would be the ideal answer to this question. "
        "Do not hedge or say you don't know.\n\n"
        f"Question: {query}"
    )
    return llm.invoke(prompt).content


def hyde_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    # Generate hypothetical document and use its embedding for retrieval
    hypo_doc = generate_hypothetical_document(query, llm)
    results = store.similarity_search(hypo_doc, k=3)

    context = "\n".join(doc.page_content for doc in results)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "HyDE generates a hypothetical answer before retrieval.",
        "Embedding a hypothetical document bridges the query-document gap.",
        "Dense retrieval works by matching embeddings in vector space.",
        "Traditional keyword search uses term frequency for ranking.",
    ]
    print(hyde_rag("How does HyDE improve retrieval?", corpus))
