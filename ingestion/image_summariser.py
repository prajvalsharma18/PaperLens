"""
Summarises chart/figure images using a vision model (GPT-4o-mini).

The summary is used as the embedding text; the raw image is stored
separately in the document store, linked by UUID.
"""

import base64
import os
from pathlib import Path
from typing import Any


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for the OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def summarise_images(
    image_elements: list[dict[str, Any]],
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call the vision model on each image element to generate a
    searchable text summary.

    Returns elements with an added 'embed_content' key containing
    the vision model's description.
    """
    from openai import OpenAI

    client = OpenAI()
    vision_model = model or os.getenv("VISION_MODEL", "gpt-4o-mini")
    summarised = []

    for el in image_elements:
        image_path = el.get("metadata", {}).get("image_path")

        if not image_path or not Path(image_path).exists():
            # No image file — embed a placeholder
            el["embed_content"] = "[Image: no extractable content]"
            summarised.append(el)
            continue

        encoded = _encode_image(image_path)
        ext = Path(image_path).suffix.lstrip(".") or "png"

        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{encoded}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image or chart in detail. "
                                "If it is a chart or graph, describe the data it shows, "
                                "including axis labels, trends, key values, and any title. "
                                "Be specific and factual."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        summary = response.choices[0].message.content.strip()
        summarised.append({
            **el,
            "embed_content": f"Figure description: {summary}",
            "vision_summary": summary,
        })

    return summarised
