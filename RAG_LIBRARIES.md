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

### 4. Haystack

**GitHub:** [deepset-ai/haystack](https://github.com/deepset-ai/haystack)

An open-source Python framework by deepset for building RAG pipelines and LLM applications. Haystack provides a pipeline-based architecture where you compose retrieval, ranking, and generation components into flexible, production-ready workflows.

**Key features:**
- Modular pipeline architecture with reusable components
- Built-in support for re-ranking, query expansion, and hybrid retrieval
- Document converters for PDF, HTML, DOCX, and more
- Evaluation tools for end-to-end RAG pipeline assessment

```bash
pip install haystack-ai
```

---

### 5. Unstructured

**GitHub:** [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured)

A Python library for preprocessing and ingesting unstructured data (PDFs, emails, images, HTML, Word docs) into formats suitable for RAG pipelines. It handles partitioning, cleaning, and chunking documents so they are ready for embedding and retrieval.

**Key features:**
- Partitioning for 25+ file types with automatic format detection
- Table extraction and OCR for scanned documents
- Built-in chunking strategies (by title, by page, overlap-aware)
- Direct integrations with LangChain, LlamaIndex, and Haystack

```bash
pip install unstructured
```

---

## Quick Comparison

| Library      | Primary Role             | Best For                                     |
|--------------|--------------------------|----------------------------------------------|
| LangChain    | RAG framework            | End-to-end pipelines with broad integrations |
| LlamaIndex   | RAG data framework       | Advanced indexing and query strategies        |
| Docling      | Document parsing         | High-fidelity ingestion of complex documents |
| Haystack     | RAG pipeline framework   | Modular, production-ready RAG workflows      |
| Unstructured | Document preprocessing   | Ingesting and chunking diverse file formats  |
