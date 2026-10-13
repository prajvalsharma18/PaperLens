"""Dedicated Qdrant storage for CLIP-compatible visual vectors."""

from __future__ import annotations

import math
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from retrieval.vector_store import _get_client


DEFAULT_VISUAL_COLLECTION = "rag_visuals"


class VisualVectorStoreError(RuntimeError):
    """Raised for visual collection configuration or Qdrant failures."""


def _unit_vector(vector: Any, vector_size: int, label: str) -> list[float]:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise VisualVectorStoreError(f"{label} contains a non-numeric value.") from exc
    if len(values) != vector_size:
        raise VisualVectorStoreError(
            f"{label} has dimension {len(values)}; expected {vector_size}."
        )
    if not all(math.isfinite(value) for value in values):
        raise VisualVectorStoreError(f"{label} contains a non-finite value.")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise VisualVectorStoreError(f"{label} has an invalid or zero magnitude.")
    return [value / norm for value in values]


def visual_collection_name() -> str:
    return os.getenv("VISUAL_COLLECTION_NAME", DEFAULT_VISUAL_COLLECTION)


def visual_point_id(element_id: str) -> str:
    """Return a stable Qdrant UUID for a parent image or child crop."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"paperlens-visual-point:{element_id}")
    )


def visual_collection_exists(client: QdrantClient | None = None) -> bool:
    client = client or _get_client()
    try:
        return bool(client.collection_exists(visual_collection_name()))
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Could not check visual Qdrant collection {visual_collection_name()!r}: {exc}"
        ) from exc


def ensure_visual_collection(
    vector_size: int,
    client: QdrantClient | None = None,
) -> QdrantClient:
    if vector_size <= 0:
        raise VisualVectorStoreError("Visual embedding dimension must be positive.")
    client = client or _get_client()
    collection = visual_collection_name()
    try:
        if not client.collection_exists(collection):
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return client
        info = client.get_collection(collection)
        configured_vectors = info.config.params.vectors
        configured_size = getattr(configured_vectors, "size", None)
        if configured_size != vector_size:
            raise VisualVectorStoreError(
                f"Visual collection {collection!r} has vector dimension "
                f"{configured_size!r}, but model dimension is {vector_size}. "
                "Choose another VISUAL_COLLECTION_NAME or explicitly manage the "
                "existing collection; it will not be altered."
            )
        return client
    except VisualVectorStoreError:
        raise
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Could not prepare visual Qdrant collection {collection!r}: {exc}"
        ) from exc


def upsert_visual_vectors(
    records: list[dict[str, Any]],
    vectors: list[list[float]],
    vector_size: int,
    client: QdrantClient | None = None,
) -> int:
    if len(records) != len(vectors):
        raise VisualVectorStoreError(
            "Visual record and embedding counts do not match."
        )
    if not records:
        return 0
    client = ensure_visual_collection(vector_size, client)
    points: list[PointStruct] = []
    for record, vector in zip(records, vectors):
        element_id = str(record.get("element_id") or record.get("chunk_id") or "")
        if not element_id:
            raise VisualVectorStoreError(
                "Visual record is missing its stable element or chunk ID."
            )
        vector = _unit_vector(vector, vector_size, f"Visual vector for {element_id!r}")
        payload = {key: value for key, value in record.items() if value is not None}
        points.append(
            PointStruct(
                id=visual_point_id(element_id),
                vector=vector,
                payload=payload,
            )
        )
    try:
        client.upsert(
            collection_name=visual_collection_name(),
            points=points,
            wait=True,
        )
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Could not upsert visual points into {visual_collection_name()!r}: {exc}"
        ) from exc
    return len(points)


def validate_visual_collection_dimension(
    vector_size: int,
    client: QdrantClient | None = None,
) -> bool:
    client = client or _get_client()
    collection = visual_collection_name()
    try:
        if not client.collection_exists(collection):
            return False
        configured_size = getattr(
            client.get_collection(collection).config.params.vectors,
            "size",
            None,
        )
        if configured_size != vector_size:
            raise VisualVectorStoreError(
                f"Visual collection {collection!r} has vector dimension "
                f"{configured_size!r}, but model dimension is {vector_size}. "
                "Choose another VISUAL_COLLECTION_NAME or explicitly manage the "
                "existing collection; it will not be altered."
            )
        return True
    except VisualVectorStoreError:
        raise
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Could not validate visual Qdrant collection {collection!r}: {exc}"
        ) from exc


def search_visual_vectors(
    query_vector: list[float],
    limit: int,
    document_id: str | None = None,
    client: QdrantClient | None = None,
) -> list[Any]:
    if limit < 1:
        raise VisualVectorStoreError("Visual search limit must be positive.")
    client = client or _get_client()
    if not visual_collection_exists(client):
        return []
    try:
        collection_info = client.get_collection(visual_collection_name())
        vector_size = getattr(collection_info.config.params.vectors, "size", None)
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Could not inspect visual Qdrant collection {visual_collection_name()!r}: {exc}"
        ) from exc
    if not isinstance(vector_size, int) or vector_size < 1:
        raise VisualVectorStoreError(
            f"Visual collection {visual_collection_name()!r} has no usable vector dimension."
        )
    query_vector = _unit_vector(query_vector, vector_size, "Visual query vector")
    query_filter = (
        Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )
        if document_id
        else None
    )
    try:
        response = client.query_points(
            collection_name=visual_collection_name(),
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        raise VisualVectorStoreError(
            f"Visual Qdrant search failed in {visual_collection_name()!r}: {exc}"
        ) from exc
    return list(response.points)
