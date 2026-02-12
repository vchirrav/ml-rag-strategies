"""Late Chunking: Embed the full document first, then chunk the embeddings."""

import os
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def late_chunk_embed(
    full_text: str, chunk_size: int = 50, embeddings_model: OpenAIEmbeddings = None
) -> list[tuple[str, list[float]]]:
    """
    Simulate late chunking: embed the full document, then split into chunks.
    Each chunk's embedding is derived from the full-document context.

    Note: True late chunking requires token-level embeddings from a model like
    jina-embeddings-v2. Here we approximate by embedding overlapping windows.
    """
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        # Embed with full document as prefix for context
        contextual_input = full_text[:200] + " ... " + chunk_text
        embedding = embeddings_model.embed_query(contextual_input)
        chunks.append((chunk_text, embedding))
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def late_chunking_rag(query: str, document: str) -> str:
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    chunk_embeddings = late_chunk_embed(document, chunk_size=30, embeddings_model=embeddings_model)
    query_embedding = embeddings_model.embed_query(query)

    # Rank chunks by cosine similarity
    scored = [
        (text, cosine_similarity(query_embedding, emb))
        for text, emb in chunk_embeddings
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [text for text, _ in scored[:3]]

    context = "\n".join(top_chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    document = (
        "Late chunking is a technique where the full document is processed by the "
        "embedding model before splitting into chunks. This preserves cross-chunk "
        "context that is lost in traditional chunk-then-embed approaches. "
        "Each chunk embedding benefits from the full document context, leading to "
        "better retrieval quality for ambiguous or context-dependent passages."
    )
    print(late_chunking_rag("What is late chunking?", document))
