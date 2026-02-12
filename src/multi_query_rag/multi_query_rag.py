"""Multi-Query RAG: Generate multiple sub-questions and retrieve for each independently."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def decompose_query(query: str, llm: ChatOpenAI) -> list[str]:
    """Break a complex query into independent sub-questions."""
    prompt = (
        "Break this question into 2-3 independent sub-questions that, "
        "when answered together, fully address the original. "
        "Return one per line, no numbering.\n\n"
        f"Question: {query}"
    )
    result = llm.invoke(prompt).content
    return [line.strip() for line in result.strip().split("\n") if line.strip()]


def multi_query_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    sub_questions = decompose_query(query, llm)

    # Retrieve relevant docs for each sub-question
    all_context = []
    for sq in sub_questions:
        results = store.similarity_search(sq, k=2)
        for doc in results:
            if doc.page_content not in all_context:
                all_context.append(doc.page_content)

    context = "\n".join(all_context)
    response = llm.invoke(
        f"Answer the original question using the context below.\n"
        f"Context:\n{context}\n\nOriginal question: {query}"
    )
    return response.content


if __name__ == "__main__":
    corpus = [
        "Vector databases enable fast similarity search over embeddings.",
        "LLMs generate text based on input prompts and context.",
        "RAG reduces hallucination by grounding responses in retrieved data.",
        "Chunking strategies affect retrieval precision and recall.",
    ]
    print(multi_query_rag("How does RAG reduce hallucination and what role do vector DBs play?", corpus))
