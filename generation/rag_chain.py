"""
Multimodal RAG generation chain.

Constructs a context-aware prompt that includes:
  - retrieved text excerpts
  - full HTML/Markdown for retrieved tables
  - the actual chart image (base64) for retrieved figures

Passes the assembled prompt to the configured LLM.
"""

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from ingestion.assets import resolve_asset_path


def _load_image_b64(image_path: str) -> str | None:
    path = resolve_asset_path(image_path)
    if not path.exists():
        return None
    try:
        image_bytes = path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("An image evidence asset is corrupt or unreadable.") from exc
    return base64.b64encode(image_bytes).decode("utf-8")


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
    image_blocks: list[ChatCompletionContentPartImageParam] = []
    labeled_image_blocks: list[
        tuple[str, ChatCompletionContentPartImageParam]
    ] = []

    for item in context:
        item_type = item.get("type", "NarrativeText")
        content = item.get("content", "")

        if item_type == "Image":
            metadata = item.get("metadata", {})
            image_path = metadata.get("image_path")
            if image_path:
                try:
                    b64 = _load_image_b64(image_path)
                except ValueError:
                    if metadata.get("require_image_evidence"):
                        raise
                    b64 = None
                    print(
                        f"Warning: image evidence {metadata.get('evidence_label', '')!r} "
                        "is corrupt; using its text summary.",
                        file=sys.stderr,
                    )
                if b64:
                    ext = Path(image_path).suffix.lstrip(".") or "png"
                    image_block: ChatCompletionContentPartImageParam = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                    }
                    label = metadata.get("evidence_label")
                    if label:
                        labeled_image_blocks.append((str(label), image_block))
                    else:
                        image_blocks.append(image_block)
                    continue
            if metadata.get("require_image_evidence"):
                label = metadata.get("evidence_label", "requested page")
                raise ValueError(
                    f"Required image evidence {label!r} is unavailable."
                )
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
        "Each labelled image is identified by its document and page; keep "
        "evidence associated with the page named in its label, especially when "
        "comparing pages. "
        "If the answer cannot be determined from the context, say so clearly. "
        "Do not hallucinate."
    )

    context_part: ChatCompletionContentPartTextParam = {
        "type": "text",
        "text": (
            "Context:\n\n"
            + "\n\n---\n\n".join(text_parts)
            + f"\n\nQuestion: {question}"
        ),
    }
    user_content: list[
        ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
    ] = [context_part]
    user_content.extend(image_blocks)
    for label, image_block in labeled_image_blocks:
        label_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": f"Image evidence: {label}",
        }
        user_content.extend([label_part, image_block])

    messages: list[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    ] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    if answer is None:
        raise ValueError("The model returned an empty answer.")
    return answer.strip()
