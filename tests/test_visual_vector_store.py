from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, VectorParams

from retrieval.visual_vector_store import (
    VisualVectorStoreError,
    ensure_visual_collection,
    search_visual_vectors,
    upsert_visual_vectors,
    validate_visual_collection_dimension,
    visual_point_id,
)


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.points = {}
        self.create_calls = []
        self.upsert_calls = []
        self.query_calls = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.create_calls.append((collection_name, vectors_config))
        self.collections[collection_name] = vectors_config

    def get_collection(self, collection_name):
        config = self.collections[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=config))
        )

    def upsert(self, collection_name, points, wait):
        self.upsert_calls.append((collection_name, points, wait))
        collection = self.points.setdefault(collection_name, {})
        for point in points:
            collection[str(point.id)] = point

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        points = list(self.points.get(kwargs["collection_name"], {}).values())
        if kwargs.get("query_filter"):
            expected_document = (
                kwargs["query_filter"].must[0].match.value
            )
            points = [
                point
                for point in points
                if point.payload.get("document_id") == expected_document
            ]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=point.id,
                    payload=point.payload,
                    score=1.0 - index / 10,
                )
                for index, point in enumerate(points[: kwargs["limit"]])
            ]
        )


def _record(element_id, document_id="report"):
    return {
        "document_id": document_id,
        "element_id": element_id,
        "asset_path": f"assets/{element_id}.png",
        "chunk_type": "image_tile",
        "bbox": [0, 0, 10, 10],
    }


def test_visual_collection_uses_model_dimension_and_cosine_distance(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "clip-vectors")
    client = FakeClient()

    ensure_visual_collection(512, client)

    config = client.collections["clip-vectors"]
    assert isinstance(config, VectorParams)
    assert config.size == 512
    assert config.distance == Distance.COSINE
    assert len(client.create_calls) == 1


def test_dimension_mismatch_fails_without_modifying_collection(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "clip-vectors")
    client = FakeClient()
    client.create_collection(
        collection_name="clip-vectors",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    with pytest.raises(VisualVectorStoreError, match="has vector dimension 384"):
        ensure_visual_collection(512, client)

    assert len(client.create_calls) == 1
    assert not client.upsert_calls
    assert validate_visual_collection_dimension(384, client)


def test_upserts_use_deterministic_ids_and_do_not_duplicate_points(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "clip-vectors")
    client = FakeClient()
    record = _record("tile-1")

    first_count = upsert_visual_vectors([record], [[1.0, 0.0, 0.0]], 3, client)
    first_id = client.upsert_calls[0][1][0].id
    second_count = upsert_visual_vectors([record], [[0.0, 1.0, 0.0]], 3, client)

    assert first_count == second_count == 1
    assert first_id == visual_point_id("tile-1")
    assert client.upsert_calls[1][1][0].id == first_id
    assert len(client.points["clip-vectors"]) == 1
    assert client.points["clip-vectors"][first_id].payload["bbox"] == [0, 0, 10, 10]


def test_visual_search_applies_document_filter_and_preserves_payload(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "clip-vectors")
    client = FakeClient()
    records = [_record("tile-a", "report-a"), _record("tile-b", "report-b")]
    upsert_visual_vectors(
        records,
        [[1.0, 0.0], [0.0, 1.0]],
        2,
        client,
    )

    results = search_visual_vectors(
        [1.0, 0.0], limit=5, document_id="report-b", client=client
    )

    assert len(results) == 1
    assert results[0].payload["element_id"] == "tile-b"
    call = client.query_calls[0]
    assert call["query"] == [1.0, 0.0]
    assert call["query_filter"].must[0].key == "document_id"
    assert call["query_filter"].must[0].match.value == "report-b"


def test_search_on_uncreated_collection_returns_no_hits(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "empty-collection")
    client = FakeClient()

    assert search_visual_vectors([0.2, 0.4], 3, client=client) == []
    assert not client.query_calls


def test_visual_upsert_rejects_invalid_vector_shapes(monkeypatch):
    monkeypatch.setenv("VISUAL_COLLECTION_NAME", "clip-vectors")
    client = FakeClient()

    with pytest.raises(VisualVectorStoreError, match="counts do not match"):
        upsert_visual_vectors([_record("tile")], [], 2, client)
    with pytest.raises(VisualVectorStoreError, match="dimension 1; expected 2"):
        upsert_visual_vectors([_record("tile")], [[1.0]], 2, client)
