"""
Multi-vector retriever.

Searches Qdrant for the top-k most relevant summaries, then
fetches the corresponding raw elements from the document store.
"""

from typing import Any
from retrieval.document_store import load_raw_element


def retrieve(
    question: str,
    embedder,
    k: int = 4,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    1. Embed the question.
    2. Search Qdrant for the top-k matching summaries.
    3. For Table and Image matches, fetch the raw element from the document store.
    4. Return a unified context list for the generation step.
    """
    from retrieval.vector_store import get_vector_store

    store = get_vector_store(embedder)
    search_filter = None
    if document_id:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    if search_filter is None:
        results = store.similarity_search(question, k=k)
    else:
        results = store.similarity_search(question, k=k, filter=search_filter)

    context = []
    for doc in results:
        doc_uuid = doc.metadata.get("uuid")
        doc_type = doc.metadata.get("type", "NarrativeText")

        if doc_uuid:
            raw = load_raw_element(doc_uuid)
            if raw:
                if raw.get("uuid") != doc_uuid:
                    raise ValueError(
                        f"Stored element UUID does not match indexed UUID {doc_uuid!r}"
                    )
                context.append(raw)
                continue

        # Plain text — use the vector store content directly
        context.append({
            "uuid": doc_uuid,
            "type": doc_type,
            "content": doc.page_content,
            "metadata": doc.metadata,
            "original_element_missing": bool(
                doc_type in ("Table", "Image") and doc_uuid
            ),
        })

    return context
