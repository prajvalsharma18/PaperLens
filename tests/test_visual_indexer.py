from types import SimpleNamespace

from PIL import Image

import retrieval.visual_indexer as visual_indexer
from ingestion.assets import asset_reference, document_asset_key, storage_root
from retrieval.document_store import save_document_manifest, save_raw_element
from retrieval.visual_indexer import (
    discover_visual_assets,
    index_document_visual_assets,
    search_visual_assets,
)


class FakeEmbedder:
    model_name = "mock/clip-test"
    dimension = 3

    def __init__(self):
        self.calls = []

    def encode_images(self, paths, batch_size):
        self.calls.append((paths, batch_size))
        return [[1.0, 0.0, 0.0] for _ in paths]

    def encode_text(self, text):
        self.calls.append(text)
        return [0.0, 1.0, 0.0]


def test_manifest_assets_discover_figure_opted_page_and_tile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    root = storage_root()
    figure_path = root / "assets" / "figures" / "figure.png"
    page_path = root / "assets" / "pages" / "page.png"
    tile_path = (
        root
        / "assets"
        / document_asset_key("report")
        / "visual_chunks"
        / "tile.png"
    )
    for path, color in (
        (figure_path, "red"),
        (page_path, "white"),
        (tile_path, "blue"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), color).save(path)
    elements = [
        {
            "uuid": "figure-1",
            "type": "Image",
            "metadata": {
                "document_id": "report",
                "page": 2,
                "source_file": "report.pdf",
                "image_path": asset_reference(figure_path),
                "bbox": [1, 2, 30, 20],
                "coordinate_system": {"origin": "bottom-left", "units": "points"},
            },
        },
        {
            "uuid": "page-1",
            "type": "Image",
            "metadata": {
                "document_id": "report",
                "page": 1,
                "source_file": "report.pdf",
                "image_path": asset_reference(page_path),
                "extraction_method": "pypdfium2_page_render",
            },
        },
    ]
    tile = {
        "uuid": "tile-1",
        "type": "ImageTile",
        "metadata": {
            "document_id": "report",
            "parent_id": "figure-1",
            "root_source_id": "figure-1",
            "source_file": "report.pdf",
            "page": 2,
            "child_image_path": asset_reference(tile_path),
            "parent_image_path": asset_reference(figure_path),
            "bbox": [0, 0, 40, 30],
            "bbox_coordinate_system": {
                "origin": "top-left",
                "units": "pixels",
                "relative_to": "parent_image",
            },
            "parent_width": 80,
            "parent_height": 60,
            "content_hash": "tile-hash",
            "chunking_method": "regular_grid_v1",
            "chunking_parameters": {"tile_width": 40, "overlap": 8},
            "source_bbox": [1, 2, 30, 20],
            "source_coordinate_system": {"origin": "bottom-left", "units": "points"},
        },
    }

    records, skipped = discover_visual_assets(
        "report",
        {"visual_chunks": [tile]},
        elements,
    )

    assert not skipped
    assert [record["chunk_type"] for record in records] == [
        "figure",
        "page_image",
        "image_tile",
    ]
    assert records[0]["bbox_coordinate_system"]["origin"] == "bottom-left"
    assert records[1]["page_number"] == 1
    assert records[2]["parent_id"] == "figure-1"
    assert records[2]["asset_path"] == asset_reference(tile_path)
    assert records[2]["parent_asset_path"] == asset_reference(figure_path)
    assert records[2]["bbox_coordinate_system"]["relative_to"] == "parent_image"
    assert records[2]["source_coordinate_system"]["origin"] == "bottom-left"


def test_indexing_uses_manifest_and_batches_images_without_openai(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    root = storage_root()
    figure_path = root / "assets" / document_asset_key("doc-1") / "extracted" / "f.png"
    tile_path = root / "assets" / document_asset_key("doc-1") / "visual_chunks" / "t.png"
    for path in (figure_path, tile_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "green").save(path)
    save_raw_element(
        {
            "uuid": "figure-parent",
            "type": "Image",
            "metadata": {
                "document_id": "doc-1",
                "page": 5,
                "image_path": asset_reference(figure_path),
            },
        }
    )
    tile = {
        "uuid": "tile-child",
        "type": "ImageTile",
        "content": "",
        "metadata": {
            "document_id": "doc-1",
            "parent_id": "figure-parent",
            "root_source_id": "figure-parent",
            "page": 5,
            "child_image_path": asset_reference(tile_path),
            "parent_image_path": asset_reference(figure_path),
            "bbox": [5, 0, 25, 24],
            "bbox_coordinate_system": {"origin": "top-left", "units": "pixels"},
            "parent_width": 32,
            "parent_height": 24,
            "content_hash": "tile",
            "chunking_method": "regular_grid_v1",
            "chunking_parameters": {"overlap": 4},
        },
    }
    save_document_manifest(
        "doc-1",
        {
            "document_id": "doc-1",
            "page_count": 5,
            "pages": [],
            "visual_chunks": [tile],
        },
    )
    embedder = FakeEmbedder()
    upserted = {}
    monkeypatch.setattr(
        visual_indexer,
        "upsert_visual_vectors",
        lambda records, vectors, dimension, client=None: upserted.update(
            records=records, vectors=vectors, dimension=dimension
        )
        or len(records),
    )

    result = index_document_visual_assets(
        "doc-1",
        batch_size=2,
        embedder=embedder,
    )

    assert result.parents_indexed == 1
    assert result.tiles_indexed == 1
    assert result.page_images_indexed == 0
    assert result.skipped_assets == 0
    assert result.model_name == "mock/clip-test"
    assert result.vector_dimension == 3
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0][0]) == 2
    assert upserted["dimension"] == 3
    assert {record["element_id"] for record in upserted["records"]} == {
        "figure-parent",
        "tile-child",
    }


def test_missing_manifest_and_invalid_assets_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    records, skipped = discover_visual_assets(
        "report",
        {"visual_chunks": [{"uuid": "missing", "metadata": {}}]},
        [
            {
                "uuid": "gone",
                "type": "Image",
                "metadata": {
                    "document_id": "report",
                    "image_path": "assets/not-there.png",
                },
            }
        ],
    )

    assert records == []
    assert len(skipped) == 2
    assert sum(skipped.values()) == 2
    try:
        index_document_visual_assets("unmanifested", embedder=FakeEmbedder())
    except ValueError as exc:
        assert "no manifest" in str(exc)
    else:
        raise AssertionError("Missing manifest must be reported.")


def test_visual_query_uses_fake_text_encoder_and_converts_results(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    image_path = tmp_path / "retrieved.png"
    Image.new("RGB", (10, 8), "orange").save(image_path)
    client = SimpleNamespace()
    monkeypatch.setattr(
        "retrieval.visual_vector_store.visual_collection_exists",
        lambda provided_client=None: True,
    )
    monkeypatch.setattr(
        visual_indexer,
        "validate_visual_collection_dimension",
        lambda dimension, provided_client=None: dimension == 3,
    )
    search_limits = []
    monkeypatch.setattr(
        visual_indexer,
        "search_visual_vectors",
        lambda vector, limit, document_id=None, client=None: search_limits.append(limit) or [
            SimpleNamespace(
                payload={
                    "element_id": "tile-1",
                    "document_id": "report",
                    "asset_path": str(image_path),
                    "parent_asset_path": "assets/parent.png",
                    "chunk_type": "image_tile",
                    "page_number": 4,
                },
                score=0.9,
            )
        ],
    )
    embedder = FakeEmbedder()

    results = search_visual_assets(
        "architecture diagram",
        limit=4,
        document_id="report",
        embedder=embedder,
        client=client,
    )

    assert embedder.calls == ["architecture diagram"]
    assert results[0]["uuid"] == "tile-1"
    assert results[0]["metadata"]["image_path"] == str(image_path)
    assert results[0]["metadata"]["page"] == 4
    assert results[0]["visual_score"] == 0.9
    assert search_limits == [32]


def test_visual_search_overfetches_then_keeps_one_hit_per_parent(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    paths = []
    points = []
    for index, (parent, score) in enumerate(
        [("parent-a", 0.99), ("parent-a", 0.98), ("parent-a", 0.97), ("parent-b", 0.96)]
    ):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (10, 8), "orange").save(path)
        paths.append(path)
        points.append(SimpleNamespace(
            payload={
                "element_id": f"tile-{index}", "document_id": "doc",
                "asset_path": str(path), "root_source_id": parent,
                "chunk_type": "image_tile", "page_number": index + 1,
            },
            score=score,
        ))
    requested_limits = []
    monkeypatch.setattr("retrieval.visual_vector_store.visual_collection_exists", lambda provided_client=None: True)
    monkeypatch.setattr(visual_indexer, "validate_visual_collection_dimension", lambda *args: True)
    monkeypatch.setattr(visual_indexer, "search_visual_vectors", lambda vector, limit, **kwargs: requested_limits.append(limit) or points)

    results = search_visual_assets("diagram", limit=2, embedder=FakeEmbedder())

    assert requested_limits == [16]
    assert [result["uuid"] for result in results] == ["tile-0", "tile-3"]
