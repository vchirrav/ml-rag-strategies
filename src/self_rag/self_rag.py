"""Self-RAG: The model self-reflects on whether retrieval is needed and if the response is grounded."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def should_retrieve(query: str, llm: ChatOpenAI) -> bool:
    """Self-reflect: does this query need external knowledge?"""
    prompt = (
        "Determine if this question requires external knowledge to answer accurately. "
        "Reply with only 'yes' or 'no'.\n\n"
        f"Question: {query}"
    )
    result = llm.invoke(prompt).content.strip().lower()
    return result.startswith("yes")


def is_response_grounded(response: str, context: str, llm: ChatOpenAI) -> bool:
    """Self-reflect: is the response supported by the retrieved context?"""
    prompt = (
        "Is the following response fully supported by the context? "
        "Reply with only 'yes' or 'no'.\n\n"
        f"Context:\n{context}\n\nResponse:\n{response}"
    )
    result = llm.invoke(prompt).content.strip().lower()
    return result.startswith("yes")


def self_rag(query: str, corpus: list[str]) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    # Step 1: Decide if retrieval is needed
    if not should_retrieve(query, llm):
        return llm.invoke(query).content

    # Step 2: Retrieve
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)
    results = store.similarity_search(query, k=3)
    context = "\n".join(doc.page_content for doc in results)

    # Step 3: Generate
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    ).content

    # Step 4: Verify groundedness
    if not is_response_grounded(response, context, llm):
        response = llm.invoke(
            f"The previous answer was not grounded. Re-answer strictly from context.\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        ).content

    return response


if __name__ == "__main__":
    corpus = [
        "Self-RAG adds reflection tokens to control retrieval and generation.",
        "The model decides when to retrieve and verifies answer groundedness.",
        "This reduces hallucination by filtering unsupported claims.",
    ]
    print(self_rag("How does Self-RAG reduce hallucination?", corpus))
