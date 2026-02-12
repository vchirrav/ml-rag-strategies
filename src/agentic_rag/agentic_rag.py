"""Agentic RAG: An LLM agent decides when and how to retrieve information."""

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Build a simple vector store
_docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation."),
    Document(page_content="Agents can plan multi-step retrieval workflows."),
    Document(page_content="LangGraph enables stateful agent orchestration."),
]
_store = FAISS.from_documents(_docs, OpenAIEmbeddings(api_key=OPENAI_API_KEY))
_retriever = _store.as_retriever(search_kwargs={"k": 2})


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = _retriever.invoke(query)
    return "\n".join(doc.page_content for doc in results)


def agentic_rag(query: str) -> str:
    """The agent autonomously decides whether to retrieve or answer directly."""
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    agent = create_react_agent(llm, tools=[search_knowledge_base])
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(agentic_rag("What is RAG and how do agents use it?"))
