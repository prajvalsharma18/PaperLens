from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from retrieval.visual_embedder import VisualEmbedder, VisualEmbeddingError


class FakeProcessor:
    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def __call__(self, *, images=None, text=None, **kwargs):
        if images is not None:
            self.image_calls.append(images)
            return {"pixel_values": images}
        self.text_calls.append(text)
        return {"input_ids": text}


class FakeModel:
    config = SimpleNamespace(projection_dim=3)

    def __init__(self, image_vectors=None, text_vectors=None):
        self.image_vectors = image_vectors or [[3.0, 4.0, 0.0]]
        self.text_vectors = text_vectors or [[0.0, 4.0, 3.0]]
        self.image_calls = []
        self.text_calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def get_image_features(self, pixel_values):
        self.image_calls.append(pixel_values)
        return SimpleNamespace(pooler_output=self.image_vectors[: len(pixel_values)])

    def get_text_features(self, input_ids):
        self.text_calls.append(input_ids)
        return self.text_vectors[: len(input_ids)]


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    class cuda:
        @staticmethod
        def is_available():
            return False


def _fake_embedder(**model_kwargs):
    processor = FakeProcessor()
    model = FakeModel(**model_kwargs)
    embedder = VisualEmbedder(
        processor=processor,
        model=model,
        torch_module=FakeTorch(),
        device="cpu",
    )
    return embedder, processor, model


def test_image_and_text_features_are_normalized_and_share_configured_dimension(
    tmp_path,
):
    image_paths = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (8, 6), (index * 80, 10, 20)).save(path)
        image_paths.append(path)
    embedder, processor, model = _fake_embedder(
        image_vectors=[[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]
    )

    vectors = embedder.encode_images(image_paths, batch_size=2)
    text_vector = embedder.encode_text("a red bar chart")

    assert embedder.dimension == 3
    assert vectors == [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
    assert text_vector == [0.0, 0.8, 0.6]
    assert len(processor.image_calls) == 1
    assert model.device == "cpu"
    assert model.image_calls
    assert processor.text_calls == [["a red bar chart"]]


def test_image_batches_are_processed_in_configured_batch_sizes(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"batch-{index}.png"
        Image.new("RGB", (2, 2), "white").save(path)
        paths.append(path)
    vectors = [[1.0, 0.0, 0.0]] * 3
    embedder, processor, _ = _fake_embedder(image_vectors=vectors)

    assert embedder.encode_images(paths, batch_size=2) == vectors
    assert [len(batch) for batch in processor.image_calls] == [2, 1]


@pytest.mark.parametrize("content", [b"", b"not a valid image"])
def test_empty_and_malformed_images_fail_without_model_inference(
    tmp_path, content
):
    path = tmp_path / "invalid.png"
    path.write_bytes(content)
    embedder, _, model = _fake_embedder()

    with pytest.raises(VisualEmbeddingError, match="not a readable image"):
        embedder.encode_image(path)

    assert not model.image_calls


def test_missing_images_and_empty_queries_are_rejected(tmp_path):
    embedder, _, _ = _fake_embedder()
    with pytest.raises(VisualEmbeddingError, match="does not exist"):
        embedder.encode_image(tmp_path / "missing.png")
    with pytest.raises(VisualEmbeddingError, match="cannot be empty"):
        embedder.encode_text("  ")


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.0, 0.0, 0.0]], "zero embedding"),
        ([[float("nan"), 0.0, 1.0]], "non-finite"),
        ([[1.0, 0.0]], "expected 3"),
    ],
)
def test_invalid_model_embeddings_are_rejected(vectors, message):
    embedder, _, _ = _fake_embedder(text_vectors=vectors)

    with pytest.raises(VisualEmbeddingError, match=message):
        embedder.encode_text("valid query")


def test_invalid_dimension_device_and_batch_configuration_fail():
    class NoDimensionModel(FakeModel):
        config = SimpleNamespace()

    with pytest.raises(VisualEmbeddingError, match="projection_dim"):
        VisualEmbedder(
            processor=FakeProcessor(),
            model=NoDimensionModel(),
            torch_module=FakeTorch(),
        )

    with pytest.raises(VisualEmbeddingError, match="CUDA is unavailable"):
        VisualEmbedder(
            processor=FakeProcessor(),
            model=FakeModel(),
            torch_module=FakeTorch(),
            device="cuda",
        )

    embedder, _, _ = _fake_embedder()
    with pytest.raises(VisualEmbeddingError, match="batch size must be positive"):
        embedder.encode_images([], batch_size=0)
