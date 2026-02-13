import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document 
from sentence_transformers import CrossEncoder

# Load API key securely from environment
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def build_vectorstore(docs: list[str]) -> FAISS:
    documents = [Document(page_content=d) for d in docs]
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.from_documents(documents, embeddings)


def rerank(query: str, candidates: list[Document], top_k: int = 3) -> list[Document]:
    """Use a cross-encoder to re-score retrieved documents."""
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def reranking_rag(query: str, corpus: list[str]) -> str:
    store = build_vectorstore(corpus)
    # Step 1: Broad retrieval (fetch more than needed)
    initial_results = store.similarity_search(query, k=10)
    # Step 2: Re-rank with cross-encoder
    top_docs = rerank(query, initial_results, top_k=3)

    context = "\n".join(doc.page_content for doc in top_docs)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    sample_corpus = [
        "Python is a high-level programming language.",
        "RAG combines retrieval with generation for better answers.",
        "Re-ranking improves retrieval by scoring candidate relevance.",
        "Machine learning models learn patterns from data.",
        "Vector databases store embeddings for similarity search.",
    ]
    print(reranking_rag("What is re-ranking in RAG?", sample_corpus))
