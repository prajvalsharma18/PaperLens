"""
Multimodal RAG generation chain.

Constructs a context-aware prompt that includes:
  - retrieved text excerpts
  - full HTML/Markdown for retrieved tables
  - the actual chart image (base64) for retrieved figures

Passes the assembled prompt to the configured LLM.
"""

import base64
import os
from pathlib import Path
from typing import Any


def _load_image_b64(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_answer(question: str, context: list[dict[str, Any]]) -> str:
    """
    Build a multimodal prompt from retrieved context and call the LLM.

    Text and table context is included as message text.
    Image context is included as base64 image_url blocks (vision input).
    """
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("LLM_MODEL", "gpt-4o")

    text_parts: list[str] = []
    image_blocks: list[dict] = []

    for item in context:
        item_type = item.get("type", "NarrativeText")
        content = item.get("content", "")

        if item_type == "Image":
            image_path = item.get("metadata", {}).get("image_path")
            if image_path:
                b64 = _load_image_b64(image_path)
                if b64:
                    ext = Path(image_path).suffix.lstrip(".") or "png"
                    image_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                    })
                    continue
            # Fallback to vision summary if image not available
            summary = item.get("vision_summary", content)
            text_parts.append(f"[Figure]: {summary}")

        elif item_type == "Table":
            text_parts.append(f"[Table]:\n{content}")

        else:
            text_parts.append(content)

    system_prompt = (
        "You are a precise document analysis assistant. "
        "Answer the user's question using only the provided context — "
        "text excerpts, tables, and chart images. "
        "If the answer cannot be determined from the context, say so clearly. "
        "Do not hallucinate."
    )

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Context:\n\n"
                + "\n\n---\n\n".join(text_parts)
                + f"\n\nQuestion: {question}"
            ),
        }
    ] + image_blocks

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
