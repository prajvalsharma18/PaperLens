# Architecture

## System Diagram

```mermaid
graph TD
    PDF(["PDF Document"])
    UNST["Unstructured.io\nElement Partitioner"]
    TEXT["NarrativeText + Titles"]
    TABLES["Tables (HTML/Markdown)"]
    IMAGES["Images / Charts"]
    VISION["Vision Model\nGPT-4o-mini"]
    EMBED["Embeddings"]
    QDRANT["Qdrant Vector Store\n(summaries)"]
    DOCSTORE["Document Store\n(raw tables + images)"]
    QUERY(["User Query"])
    RETRIEVER["Multi-Vector Retriever"]
    LLM["Multimodal LLM\nGPT-4o"]
    ANSWER(["Answer"])
    RAGAS["RAGAS Evaluator"]

    PDF --> UNST
    UNST --> TEXT
    UNST --> TABLES
    UNST --> IMAGES
    IMAGES --> VISION
    VISION -->|"summary"| EMBED
    TABLES --> EMBED
    TEXT --> EMBED
    EMBED --> QDRANT
    TABLES --> DOCSTORE
    IMAGES --> DOCSTORE
    QUERY --> RETRIEVER
    RETRIEVER --> QDRANT
    QDRANT -->|"UUID"| DOCSTORE
    DOCSTORE -->|"raw data"| LLM
    RETRIEVER -->|"text chunks"| LLM
    LLM --> ANSWER
    ANSWER --> RAGAS
```

## Module Responsibilities

| Module | File | Responsibility |
|---|---|---|
| Partitioner | `ingestion/partitioner.py` | PDF element extraction (Unstructured.io) |
| Table Converter | `ingestion/table_converter.py` | Table → HTML/Markdown |
| Image Summariser | `ingestion/image_summariser.py` | Vision model chart description |
| Embedder | `retrieval/embedder.py` | Embedding model wrapper |
| Vector Store | `retrieval/vector_store.py` | Qdrant index management |
| Document Store | `retrieval/document_store.py` | Raw element persistence (UUID-keyed) |
| Multi-Vector Retriever | `retrieval/multi_vector_retriever.py` | Summary search → raw data fetch |
| RAG Chain | `generation/rag_chain.py` | Multimodal prompt construction + LLM |
| RAGAS Evaluator | `evaluation/ragas_evaluator.py` | Faithfulness, Relevancy, Precision |
| Benchmark | `evaluation/benchmark.py` | Multimodal vs text-only comparison |
