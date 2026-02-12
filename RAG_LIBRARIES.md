# Top 5 Popular Libraries for RAG

A curated list of the most widely-used libraries for building Retrieval-Augmented Generation (RAG) pipelines.

---

### 1. LangChain

**GitHub:** [langchain-ai/langchain](https://github.com/langchain-ai/langchain)

A framework for developing LLM-powered applications. LangChain provides modular components for document loading, text splitting, embedding, vector storage, retrieval, and generation — making it one of the most popular choices for end-to-end RAG pipelines.

**Key features:**
- 160+ document loaders (PDF, HTML, Markdown, databases, APIs)
- Built-in text splitters with overlap control
- Integrations with 50+ vector stores and embedding providers
- Chains and agents for complex retrieval workflows

```bash
pip install langchain langchain-openai langchain-community
```

---

### 2. LlamaIndex

**GitHub:** [run-llama/llama_index](https://github.com/run-llama/llama_index)

A data framework purpose-built for RAG. LlamaIndex specializes in indexing, structuring, and querying private data with LLMs, offering advanced retrieval strategies like hierarchical indexing, knowledge graphs, and query routing out of the box.

**Key features:**
- Specialized index types (vector, list, tree, keyword, knowledge graph)
- Built-in query engines with response synthesis
- Advanced retrieval modes (recursive, fusion, auto-merging)
- Evaluation framework for measuring retrieval and generation quality

```bash
pip install llama-index
```

---

### 3. Docling

**GitHub:** [DS4SD/docling](https://github.com/DS4SD/docling)

A document parsing library by IBM that converts complex file formats (PDF, DOCX, PPTX, XLSX, HTML) into clean, structured representations. Docling handles tables, figures, and layout-aware parsing, making it essential for high-quality document ingestion in RAG pipelines.

**Key features:**
- Advanced PDF parsing with OCR and table structure recognition
- Layout-aware document understanding
- Export to Markdown, JSON, and Docling Document format
- Integrates with LangChain and LlamaIndex for downstream RAG

```bash
pip install docling
```

---

### 4. ChromaDB

**GitHub:** [chroma-core/chroma](https://github.com/chroma-core/chroma)

An open-source embedding database designed for AI applications. Chroma provides a simple API for storing, searching, and filtering embeddings, serving as the vector store backbone for many RAG systems.

**Key features:**
- Simple Python-native API with minimal setup
- Automatic embedding generation via built-in integrations
- Metadata filtering and multi-modal support
- Runs in-memory, on-disk, or as a client-server deployment

```bash
pip install chromadb
```

---

### 5. FAISS (Facebook AI Similarity Search)

**GitHub:** [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

A high-performance library by Meta for dense vector similarity search and clustering. FAISS is the go-to choice when you need efficient nearest-neighbor retrieval at scale, powering the vector search layer in many production RAG systems.

**Key features:**
- Optimized for billion-scale vector search (GPU and CPU)
- Multiple index types (flat, IVF, HNSW, PQ) for speed/accuracy trade-offs
- Batch search and range search capabilities
- Widely used as the backend for higher-level RAG frameworks

```bash
pip install faiss-cpu   # or faiss-gpu for GPU support
```

---

## Quick Comparison

| Library    | Primary Role           | Best For                                    |
|------------|------------------------|---------------------------------------------|
| LangChain  | RAG framework          | End-to-end pipelines with broad integrations|
| LlamaIndex | RAG data framework     | Advanced indexing and query strategies       |
| Docling    | Document parsing       | High-fidelity ingestion of complex documents|
| ChromaDB   | Vector database        | Lightweight embedding storage and retrieval  |
| FAISS      | Vector similarity search | High-performance search at scale           |
