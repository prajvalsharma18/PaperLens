"""Local CLIP image/text embeddings for visual document retrieval."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


DEFAULT_VISUAL_MODEL = "openai/clip-vit-base-patch32"


class VisualEmbeddingError(RuntimeError):
    """Raised when a visual model or image cannot produce a valid embedding."""


class VisualEmbedder:
    """Reusable CLIP-compatible processor/model adapter.

    Dependencies and model weights are loaded only when an instance is created.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        processor: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "VISUAL_EMBED_MODEL", DEFAULT_VISUAL_MODEL
        )
        self.device = device or os.getenv("VISUAL_DEVICE", "cpu")
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise VisualEmbeddingError(
                    "Visual embeddings require PyTorch. Install the project "
                    "requirements in .venv."
                ) from exc
        self._torch = torch_module
        if self.device.startswith("cuda") and not self._torch.cuda.is_available():
            raise VisualEmbeddingError(
                "VISUAL_DEVICE requests CUDA, but CUDA is unavailable. "
                "Use VISUAL_DEVICE=cpu or install a CUDA-enabled PyTorch build."
            )

        if processor is None or model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                processor = processor or CLIPProcessor.from_pretrained(self.model_name)
                model = model or CLIPModel.from_pretrained(self.model_name)
            except Exception as exc:
                raise VisualEmbeddingError(
                    f"Could not load visual embedding model {self.model_name!r}: {exc}"
                ) from exc

        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        config = getattr(self.model, "config", None)
        dimension = getattr(config, "projection_dim", None)
        if not isinstance(dimension, int) or dimension <= 0:
            raise VisualEmbeddingError(
                "The configured vision-language model does not expose a positive "
                "config.projection_dim. Use a compatible CLIPModel."
            )
        self.dimension = dimension

    def _model_inputs(self, inputs: Any) -> dict[str, Any]:
        if hasattr(inputs, "to"):
            moved = inputs.to(self.device)
            return dict(moved)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }

    @staticmethod
    def _projected_features(output: Any) -> Any:
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, tuple):
            return output[0]
        return output

    def _normalize_batch(self, output: Any, expected_count: int) -> list[list[float]]:
        features = self._projected_features(output)
        if hasattr(features, "detach"):
            rows = features.detach().to("cpu").tolist()
        else:
            rows = features
        if expected_count == 1 and rows and isinstance(rows[0], (float, int)):
            rows = [rows]
        if not isinstance(rows, (list, tuple)) or len(rows) != expected_count:
            raise VisualEmbeddingError(
                f"Visual model returned {len(rows) if hasattr(rows, '__len__') else 'an invalid number of'} "
                f"vectors; expected {expected_count}."
            )

        normalized: list[list[float]] = []
        for row in rows:
            try:
                vector = [float(value) for value in row]
            except (TypeError, ValueError) as exc:
                raise VisualEmbeddingError(
                    "Visual model returned a non-numeric embedding."
                ) from exc
            if len(vector) != self.dimension:
                raise VisualEmbeddingError(
                    f"Visual model returned dimension {len(vector)}; expected "
                    f"{self.dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise VisualEmbeddingError(
                    "Visual model returned a non-finite embedding."
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0:
                raise VisualEmbeddingError("Visual model returned a zero embedding.")
            normalized.append([value / norm for value in vector])
        return normalized

    def encode_images(
        self,
        image_paths: list[str | Path],
        batch_size: int = 16,
    ) -> list[list[float]]:
        if batch_size < 1:
            raise VisualEmbeddingError("Visual embedding batch size must be positive.")
        if not image_paths:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(image_paths), batch_size):
            images: list[Image.Image] = []
            try:
                for raw_path in image_paths[start : start + batch_size]:
                    path = Path(raw_path).expanduser()
                    if not path.is_file():
                        raise VisualEmbeddingError(
                            f"Visual asset does not exist: {path}"
                        )
                    try:
                        with Image.open(path) as source:
                            source.verify()
                        with Image.open(path) as source:
                            images.append(source.convert("RGB"))
                    except (OSError, UnidentifiedImageError) as exc:
                        raise VisualEmbeddingError(
                            f"Visual asset is not a readable image: {path}"
                        ) from exc
                inputs = self._model_inputs(
                    self.processor(images=images, return_tensors="pt")
                )
                with self._torch.inference_mode():
                    output = self.model.get_image_features(**inputs)
                vectors.extend(
                    self._normalize_batch(output, expected_count=len(images))
                )
            except VisualEmbeddingError:
                raise
            except Exception as exc:
                raise VisualEmbeddingError(
                    f"Could not encode visual image batch: {exc}"
                ) from exc
            finally:
                for image in images:
                    image.close()
        return vectors

    def encode_image(self, image_path: str | Path) -> list[float]:
        return self.encode_images([image_path])[0]

    def encode_text(self, text: str) -> list[float]:
        if not text.strip():
            raise VisualEmbeddingError("Visual search query cannot be empty.")
        try:
            inputs = self._model_inputs(
                self.processor(text=[text], return_tensors="pt", padding=True)
            )
            with self._torch.inference_mode():
                output = self.model.get_text_features(**inputs)
            return self._normalize_batch(output, expected_count=1)[0]
        except VisualEmbeddingError:
            raise
        except Exception as exc:
            raise VisualEmbeddingError(
                f"Could not encode visual search query: {exc}"
            ) from exc


_VISUAL_EMBEDDER: VisualEmbedder | None = None


def get_visual_embedder() -> VisualEmbedder:
    """Return a reusable process-local visual embedder."""
    global _VISUAL_EMBEDDER
    if _VISUAL_EMBEDDER is None:
        _VISUAL_EMBEDDER = VisualEmbedder()
    return _VISUAL_EMBEDDER
