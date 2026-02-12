"""Context-Aware Chunking: Split documents using semantic boundaries, not fixed sizes."""

import os
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def semantic_chunk(text: str) -> list[str]:
    """Split text by semantic boundaries (headings, paragraphs, topic shifts)."""
    # Split on markdown-style headings or double newlines
    sections = re.split(r"\n#{1,3}\s+|\n\n+", text)
    chunks = [s.strip() for s in sections if s.strip()]
    return chunks


def chunk_with_overlap(chunks: list[str], overlap_sentences: int = 1) -> list[str]:
    """Add sentence overlap between adjacent chunks for continuity."""
    result = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if i > 0:
            prev_sentences = chunks[i - 1].split(". ")
            overlap = ". ".join(prev_sentences[-overlap_sentences:])
            prefix = overlap + ". " if not overlap.endswith(".") else overlap + " "
        result.append(prefix + chunk)
    return result


def context_aware_chunking_rag(query: str, document: str) -> str:
    raw_chunks = semantic_chunk(document)
    enriched_chunks = chunk_with_overlap(raw_chunks)

    documents = [Document(page_content=c) for c in enriched_chunks]
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    store = FAISS.from_documents(documents, embeddings)

    results = store.similarity_search(query, k=2)
    context = "\n".join(doc.page_content for doc in results)

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    doc = (
        "# Introduction\n"
        "RAG systems combine retrieval with generation.\n\n"
        "# Chunking\n"
        "Context-aware chunking respects document structure. "
        "It avoids splitting mid-sentence or mid-paragraph.\n\n"
        "# Benefits\n"
        "Better chunks lead to more relevant retrieval results."
    )
    print(context_aware_chunking_rag("How does chunking affect RAG?", doc))
