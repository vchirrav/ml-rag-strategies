"""Corrective RAG (CRAG): Evaluate retrieved documents and correct retrieval if needed."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def grade_documents(
    query: str, documents: list[Document], llm: ChatOpenAI
) -> tuple[list[Document], list[Document]]:
    """Grade each document as relevant or irrelevant to the query."""
    relevant, irrelevant = [], []
    for doc in documents:
        prompt = (
            "Is this document relevant to the question? Reply 'yes' or 'no'.\n\n"
            f"Question: {query}\nDocument: {doc.page_content}"
        )
        grade = llm.invoke(prompt).content.strip().lower()
        if grade.startswith("yes"):
            relevant.append(doc)
        else:
            irrelevant.append(doc)
    return relevant, irrelevant


def web_search_fallback(query: str) -> list[Document]:
    """Simulate a web search fallback when retrieval quality is low."""
    # In production, use a real search API (e.g., Tavily, SerpAPI)
    return [
        Document(page_content=f"[Web result] Corrective RAG evaluates and fixes retrieval for: {query}")
    ]


def corrective_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)
    retrieved = store.similarity_search(query, k=4)

    # Grade retrieved documents
    relevant, _ = grade_documents(query, retrieved, llm)

    # If too few relevant docs, fall back to web search
    if len(relevant) < 2:
        web_results = web_search_fallback(query)
        relevant.extend(web_results)

    context = "\n".join(doc.page_content for doc in relevant)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "CRAG evaluates retrieval quality before generation.",
        "If documents are irrelevant, CRAG triggers a web search.",
        "Document grading uses an LLM to assess relevance.",
        "Unrelated filler text about cooking recipes.",
    ]
    print(corrective_rag("How does CRAG handle bad retrieval?", corpus))
