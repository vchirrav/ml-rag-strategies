"""Adaptive RAG: Route queries to different retrieval strategies based on complexity."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def classify_query(query: str, llm: ChatOpenAI) -> str:
    """Classify query complexity to route to the right strategy."""
    prompt = (
        "Classify this query's complexity. Reply with exactly one of: "
        "'simple', 'moderate', 'complex'.\n\n"
        f"Query: {query}"
    )
    return llm.invoke(prompt).content.strip().lower()


def simple_retrieval(query: str, store: FAISS) -> list[Document]:
    """Direct vector search for simple factual queries."""
    return store.similarity_search(query, k=2)


def moderate_retrieval(query: str, store: FAISS, llm: ChatOpenAI) -> list[Document]:
    """Query expansion + retrieval for moderate queries."""
    expanded = llm.invoke(
        f"Rephrase this query for better search results:\n{query}"
    ).content
    results_original = store.similarity_search(query, k=2)
    results_expanded = store.similarity_search(expanded, k=2)
    seen = set()
    combined = []
    for doc in results_original + results_expanded:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
    return combined


def complex_retrieval(query: str, store: FAISS, llm: ChatOpenAI) -> list[Document]:
    """Decompose + multi-step retrieval for complex queries."""
    sub_q_prompt = (
        "Break into 2 sub-questions, one per line, no numbering:\n" + query
    )
    sub_questions = llm.invoke(sub_q_prompt).content.strip().split("\n")
    all_docs = []
    seen = set()
    for sq in sub_questions:
        sq = sq.strip()
        if not sq:
            continue
        for doc in store.similarity_search(sq, k=2):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return all_docs


def adaptive_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    complexity = classify_query(query, llm)

    if complexity == "simple":
        results = simple_retrieval(query, store)
    elif complexity == "moderate":
        results = moderate_retrieval(query, store, llm)
    else:
        results = complex_retrieval(query, store, llm)

    context = "\n".join(doc.page_content for doc in results)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "Adaptive RAG routes queries based on their complexity.",
        "Simple queries use direct retrieval; complex ones use decomposition.",
        "Query classification determines the retrieval strategy.",
        "Multi-step retrieval handles multi-hop reasoning questions.",
    ]
    print(adaptive_rag("How does adaptive RAG choose a retrieval strategy?", corpus))
