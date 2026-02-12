"""Fine-Tuned RAG: Fine-tune the embedding model or LLM on domain-specific data."""

import os
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def prepare_fine_tuning_data(
    qa_pairs: list[dict], output_path: str = "ft_data.jsonl"
) -> str:
    """Prepare JSONL training data for OpenAI fine-tuning."""
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as f:
        for pair in qa_pairs:
            entry = {
                "messages": [
                    {"role": "system", "content": "Answer using provided context only."},
                    {
                        "role": "user",
                        "content": f"Context: {pair['context']}\n\nQuestion: {pair['question']}",
                    },
                    {"role": "assistant", "content": pair["answer"]},
                ]
            }
            f.write(json.dumps(entry) + "\n")
    return str(output)


def fine_tuned_rag(query: str, corpus: list[str], model_name: str = "gpt-4o-mini") -> str:
    """RAG with a (potentially fine-tuned) model for generation."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    documents = [Document(page_content=d) for d in corpus]
    store = FAISS.from_documents(documents, embeddings)

    results = store.similarity_search(query, k=3)
    context = "\n".join(doc.page_content for doc in results)

    # Use fine-tuned model ID here (e.g., "ft:gpt-4o-mini:org::id")
    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Answer based on context only.\nContext:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    # Example: prepare training data
    training_data = [
        {
            "context": "Fine-tuning adapts a pre-trained model to a specific domain.",
            "question": "What is fine-tuning?",
            "answer": "Fine-tuning adapts a pre-trained model to perform better on domain-specific tasks.",
        },
    ]
    path = prepare_fine_tuning_data(training_data)
    print(f"Training data saved to: {path}")

    corpus = [
        "Fine-tuning adapts a pre-trained model to a specific domain.",
        "Domain-specific embeddings improve retrieval accuracy.",
        "RAG with fine-tuned models produces more accurate answers.",
    ]
    print(fine_tuned_rag("How does fine-tuning help RAG?", corpus))
