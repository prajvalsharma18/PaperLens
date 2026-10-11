"""
Qdrant vector store management.

Each document is stored with its UUID so the document store
can be queried for the corresponding raw element.
"""

import os
from typing import Any
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


def get_vector_store(embedder) -> QdrantVectorStore:
    collection = os.getenv("COLLECTION_NAME", "rag_documents")
    client = _get_client()

    # Create collection if it does not exist
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    return QdrantVectorStore(client=client, collection_name=collection, embedding=embedder)


def upsert_documents(docs: list[dict[str, Any]], embedder) -> None:
    """Embed and upsert documents into Qdrant."""
    from langchain_core.documents import Document

    store = get_vector_store(embedder)
    lc_docs = [
        Document(
            page_content=doc.get("embed_content") or doc.get("content", ""),
            metadata={
                **doc.get("metadata", {}),
                "uuid": doc["uuid"],
                "type": doc["type"],
            },
        )
        for doc in docs
    ]
    store.add_documents(lc_docs)
