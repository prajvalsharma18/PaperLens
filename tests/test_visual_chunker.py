import hashlib
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ingestion.assets import document_asset_key, resolve_asset_path
from ingestion.visual_chunker import VisualChunkError, chunk_image


def _parent_metadata(document_id="doc-1", parent_id="figure-1", **extra):
    return {
        "document_id": document_id,
        "element_id": parent_id,
        "uuid": parent_id,
        "source_file": "paper.pdf",
        "source_path": r"C:\research files\paper.pdf",
        "page": 7,
        "extraction_method": "unstructured_hi_res",
        "bbox": [10, 20, 400, 500],
        "coordinate_system": {
            "origin": "bottom-left",
            "units": "points",
        },
        **extra,
    }


def _write_quadrants(path: Path, mode="RGB"):
    image = Image.new(mode, (95, 65), 0 if mode == "L" else (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 47, 31), fill=20 if mode == "L" else (255, 0, 0))
    draw.rectangle((48, 0, 94, 31), fill=80 if mode == "L" else (0, 255, 0))
    draw.rectangle((0, 32, 47, 64), fill=140 if mode == "L" else (0, 0, 255))
    draw.rectangle((48, 32, 94, 64), fill=220 if mode == "L" else (255, 255, 0))
    image.save(path)
    image.close()


def _chunk(path, metadata, **kwargs):
    return chunk_image(
        path,
        metadata,
        tile_width=40,
        tile_height=30,
        overlap=10,
        min_image_size=20,
        **kwargs,
    )


def test_small_image_remains_the_parent_visual_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    image_path = tmp_path / "small image.png"
    Image.new("RGB", (32, 24), "red").save(image_path)

    result = chunk_image(
        image_path,
        _parent_metadata(),
        tile_width=40,
        tile_height=30,
        overlap=4,
        min_image_size=40,
    )

    assert not result.split
    assert result.chunks == []
    assert (result.parent_width, result.parent_height) == (32, 24)
    assert not list((tmp_path / "storage").rglob("*.png"))


def test_non_divisible_image_has_covered_overlapping_edge_tiles(
    tmp_path, monkeypatch
):
    storage = tmp_path / "storage with spaces"
    monkeypatch.setenv("DOC_STORE_PATH", str(storage))
    parent = tmp_path / "parent image.png"
    _write_quadrants(parent)

    result = _chunk(parent, _parent_metadata(image_path=str(parent)))

    boxes = [chunk["metadata"]["bbox"] for chunk in result.chunks]
    assert boxes == [
        [0, 0, 40, 30],
        [30, 0, 70, 30],
        [60, 0, 95, 30],
        [0, 20, 40, 50],
        [30, 20, 70, 50],
        [60, 20, 95, 50],
        [0, 40, 40, 65],
        [30, 40, 70, 65],
        [60, 40, 95, 65],
    ]
    assert all(0 <= x0 < x1 <= 95 and 0 <= y0 < y1 <= 65 for x0, y0, x1, y1 in boxes)
    assert boxes[0][2] - boxes[1][0] == 10
    assert boxes[0][3] - boxes[3][1] == 10
    assert boxes[2][2] == result.parent_width
    assert boxes[-1][3] == result.parent_height

    first = result.chunks[0]
    metadata = first["metadata"]
    assert first["type"] == "ImageTile"
    assert metadata["document_id"] == "doc-1"
    assert metadata["parent_id"] == "figure-1"
    assert metadata["root_source_id"] == "figure-1"
    assert metadata["source_file"] == "paper.pdf"
    assert metadata["source_path"] == r"C:\research files\paper.pdf"
    assert metadata["page"] == 7
    assert metadata["parent_type"] == "figure"
    assert metadata["parent_image_path"] == str(parent)
    assert metadata["crop_width"] == 40
    assert metadata["crop_height"] == 30
    assert metadata["bbox_coordinate_system"] == {
        "origin": "top-left",
        "units": "pixels",
        "convention": "half-open",
        "relative_to": "parent_image",
    }
    assert metadata["source_bbox"] == [10, 20, 400, 500]
    assert metadata["source_coordinate_system"]["origin"] == "bottom-left"

    for chunk in result.chunks:
        path = resolve_asset_path(chunk["metadata"]["child_image_path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == chunk["metadata"]["content_hash"]

    with Image.open(resolve_asset_path(first["metadata"]["child_image_path"])) as tile:
        assert tile.mode == "RGB"
        assert tile.getpixel((5, 5)) == (255, 0, 0)
    with Image.open(resolve_asset_path(result.chunks[1]["metadata"]["child_image_path"])) as tile:
        assert tile.getpixel((25, 5)) == (0, 255, 0)


@pytest.mark.parametrize(
    ("mode", "color"),
    [
        ("RGB", (0, 0, 255)),
        ("RGBA", (0, 0, 255)),
        ("L", (140, 140, 140)),
    ],
)
def test_supported_parent_modes_produce_consistent_rgb_tiles(
    tmp_path, monkeypatch, mode, color
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / f"{mode} parent.png"
    _write_quadrants(parent, mode)

    result = _chunk(parent, _parent_metadata(parent_id=f"{mode}-figure"))

    assert result.chunks
    with Image.open(resolve_asset_path(result.chunks[3]["metadata"]["child_image_path"])) as tile:
        assert tile.mode == "RGB"
        assert tile.getpixel((5, 25)) == color


def test_transparent_pixels_are_composited_on_white(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "alpha.png"
    image = Image.new("RGBA", (60, 30), (0, 0, 0, 0))
    image.save(parent)

    result = _chunk(parent, _parent_metadata())
    with Image.open(resolve_asset_path(result.chunks[0]["metadata"]["child_image_path"])) as tile:
        assert tile.getpixel((5, 5)) == (255, 255, 255)


def test_identical_inputs_reuse_child_assets_and_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "figure.png"
    _write_quadrants(parent)
    metadata = _parent_metadata(image_path=str(parent))

    first = _chunk(parent, metadata)
    second = _chunk(parent, metadata)

    assert [item["uuid"] for item in first.chunks] == [
        item["uuid"] for item in second.chunks
    ]
    assert first.chunks == second.chunks
    assert first.assets_reused == 0
    assert second.assets_reused == len(second.chunks)
    assert len(
        list(
            (
                tmp_path
                / "storage"
                / "assets"
                / document_asset_key("doc-1")
                / "visual_chunks"
            ).glob("*.png")
        )
    ) == len(first.chunks)
    assert hashlib.sha256(parent.read_bytes()).hexdigest()


def test_corrupt_cached_child_asset_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "figure.png"
    _write_quadrants(parent)
    metadata = _parent_metadata()
    first = _chunk(parent, metadata)
    cached_path = resolve_asset_path(first.chunks[0]["metadata"]["child_image_path"])
    cached_path.write_bytes(b"corrupt cached tile")

    second = _chunk(parent, metadata)

    assert second.chunks[0]["uuid"] == first.chunks[0]["uuid"]
    assert second.chunks[0]["metadata"]["content_hash"] == hashlib.sha256(
        cached_path.read_bytes()
    ).hexdigest()
    assert second.assets_reused == len(second.chunks) - 1


def test_ids_separate_distinct_parents_and_chunk_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "figure.png"
    _write_quadrants(parent)

    first = _chunk(parent, _parent_metadata(parent_id="figure-1"))
    other_parent = _chunk(parent, _parent_metadata(parent_id="figure-2"))
    other_geometry = chunk_image(
        parent,
        _parent_metadata(parent_id="figure-1"),
        tile_width=45,
        tile_height=35,
        overlap=5,
        min_image_size=20,
    )

    assert first.chunks[0]["uuid"] != other_parent.chunks[0]["uuid"]
    assert first.chunks[0]["uuid"] != other_geometry.chunks[0]["uuid"]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"tile_width": 0}, "positive integers"),
        ({"tile_height": -1}, "positive integers"),
        ({"overlap": -1}, "cannot be negative"),
        ({"overlap": 30}, "smaller than both tile dimensions"),
        ({"min_image_size": 0}, "positive integer"),
        ({"tile_width": 40.5}, "must be integers"),
    ],
)
def test_invalid_chunk_configuration_is_rejected(tmp_path, monkeypatch, settings, message):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "figure.png"
    Image.new("RGB", (90, 60), "red").save(parent)
    parameters = {
        "tile_width": 40,
        "tile_height": 30,
        "overlap": 5,
        "min_image_size": 20,
    }
    parameters.update(settings)

    with pytest.raises(VisualChunkError, match=message):
        chunk_image(parent, _parent_metadata(), **parameters)


@pytest.mark.parametrize("bad_content", [b"", b"not an image"])
def test_invalid_or_corrupt_parent_images_fail_actionably(
    tmp_path, monkeypatch, bad_content
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "storage"))
    parent = tmp_path / "bad image.png"
    parent.write_bytes(bad_content)

    with pytest.raises(VisualChunkError, match="Could not read parent image"):
        _chunk(parent, _parent_metadata())


def test_missing_image_and_invalid_output_directory_fail(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("DOC_STORE_PATH", str(storage))
    metadata = _parent_metadata()

    with pytest.raises(VisualChunkError, match="does not exist"):
        _chunk(tmp_path / "missing.png", metadata)

    parent = tmp_path / "image.png"
    Image.new("RGB", (90, 60), "red").save(parent)
    with pytest.raises(VisualChunkError, match="under DOC_STORE_PATH"):
        chunk_image(
            parent,
            metadata,
            tile_width=40,
            tile_height=30,
            overlap=5,
            min_image_size=20,
            output_directory=tmp_path / "outside-storage",
        )
