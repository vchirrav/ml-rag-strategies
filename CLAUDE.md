# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A collection of 15 Retrieval-Augmented Generation (RAG) strategy implementations in Python. This is an educational/reference codebase — there is no build system, test suite, or CI/CD pipeline.

## Running a Strategy

Each strategy is a standalone Python module with a `__main__` block:

```bash
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
python src/<strategy_dir>/<strategy>.py
```

All strategies require a valid `OPENAI_API_KEY` environment variable.

## Architecture

### Module Structure

Every strategy in `src/` follows the same pattern:
- One main entry function (e.g., `reranking_rag(query, corpus)`)
- Helper functions for the specific technique
- A runnable `__main__` block with sample data and sample corpus

### Common Stack

- **LLM:** `ChatOpenAI` with `gpt-4o-mini`
- **Embeddings:** `OpenAIEmbeddings`
- **Vector Store:** FAISS (CPU)
- **Documents:** LangChain `Document` schema
- **Agent orchestration:** LangGraph (used in agentic/corrective/adaptive strategies)

### The 15 Strategies (`src/`)

| # | Directory | Technique |
|---|-----------|-----------|
| 1 | `reranking/` | Bi-encoder retrieval + cross-encoder re-scoring |
| 2 | `agentic_rag/` | LangGraph ReAct agent with retrieval tools |
| 3 | `knowledge_graph_rag/` | NetworkX graph traversal for multi-hop reasoning |
| 4 | `contextual_retrieval/` | Prepends document-level context to chunks before embedding |
| 5 | `query_expansion/` | Generates multiple query variants for broader retrieval |
| 6 | `multi_query_rag/` | Decomposes complex questions into sub-questions |
| 7 | `context_aware_chunking/` | Splits on semantic boundaries (headings, paragraphs) |
| 8 | `late_chunking/` | Embeds full document first, then chunks embeddings |
| 9 | `hierarchical_rag/` | Two-level index: summaries (L1) → details (L2) |
| 10 | `fine_tuned_rag/` | Fine-tunes on domain QA pairs (includes JSONL prep) |
| 11 | `hyde_rag/` | Generates hypothetical answer as retrieval query (HyDE) |
| 12 | `fusion_rag/` | Multiple queries + Reciprocal Rank Fusion (RRF) |
| 13 | `self_rag/` | Self-reflection on retrieval need and groundedness |
| 14 | `corrective_rag/` | LLM grades docs; falls back to web search if low quality |
| 15 | `adaptive_rag/` | Routes queries by complexity (simple/moderate/complex) |

### Architecture Diagrams

SVG diagrams for each strategy live in `architecture/` (e.g., `01_reranking.svg`).

## Conventions for New Strategies

- Follow the existing module structure: main function, helpers, `__main__` example
- Use `os.environ["OPENAI_API_KEY"]` — never hardcode secrets
- Use LangChain `Document` schema for all document handling
- Include type hints and docstrings
- Add a corresponding SVG diagram in `architecture/`
- Update `README.md` with the new strategy description
