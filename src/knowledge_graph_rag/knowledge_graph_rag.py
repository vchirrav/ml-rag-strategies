"""Knowledge Graph RAG: Structure documents as a graph and traverse for retrieval."""

import os
import networkx as nx
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def build_knowledge_graph() -> nx.DiGraph:
    """Build a simple knowledge graph from structured triples."""
    graph = nx.DiGraph()
    triples = [
        ("RAG", "uses", "Retrieval"),
        ("RAG", "uses", "Generation"),
        ("Retrieval", "method", "Vector Search"),
        ("Retrieval", "method", "Keyword Search"),
        ("Generation", "powered_by", "LLM"),
        ("LLM", "example", "GPT-4"),
        ("Vector Search", "uses", "Embeddings"),
    ]
    for subj, rel, obj in triples:
        graph.add_edge(subj, obj, relation=rel)
    return graph


def retrieve_subgraph(graph: nx.DiGraph, entity: str, depth: int = 2) -> str:
    """Retrieve facts by traversing neighbors up to a given depth."""
    if entity not in graph:
        return f"Entity '{entity}' not found in knowledge graph."

    facts = []
    visited = set()
    queue = [(entity, 0)]
    while queue:
        node, d = queue.pop(0)
        if node in visited or d > depth:
            continue
        visited.add(node)
        for _, neighbor, data in graph.edges(node, data=True):
            facts.append(f"{node} --[{data['relation']}]--> {neighbor}")
            queue.append((neighbor, d + 1))
    return "\n".join(facts)


def knowledge_graph_rag(query: str, entity: str) -> str:
    graph = build_knowledge_graph()
    context = retrieve_subgraph(graph, entity)

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Answer using the knowledge graph facts below.\n"
        f"Facts:\n{context}\n\nQuestion: {query}"
    )
    return response.content


if __name__ == "__main__":
    print(knowledge_graph_rag("How does RAG use retrieval?", "RAG"))
