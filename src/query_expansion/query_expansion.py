"""Query Expansion: Rewrite the user query into multiple variants for broader retrieval."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def expand_query(query: str, llm: ChatOpenAI, n: int = 3) -> list[str]:
    """Generate multiple rephrasings of the original query."""
    prompt = (
        f"Generate {n} alternative phrasings of this search query. "
        f"Return one per line, no numbering.\n\nQuery: {query}"
    )
    result = llm.invoke(prompt).content
    variants = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return [query] + variants[:n]


def query_expansion_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    # Expand query into variants
    queries = expand_query(query, llm)

    # Retrieve for each variant and deduplicate
    seen = set()
    all_results = []
    for q in queries:
        for doc in store.similarity_search(q, k=3):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_results.append(doc)

    context = "\n".join(doc.page_content for doc in all_results)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "Query expansion rewrites questions to improve recall.",
        "Synonyms and paraphrases help find more relevant documents.",
        "Embedding models map text to dense vector representations.",
        "RAG pipelines combine retrieval with language model generation.",
    ]
    print(query_expansion_rag("How does query rewriting help search?", corpus))
