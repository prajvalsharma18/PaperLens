"""Tests for multimodal prompt construction without OpenAI requests."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image


def _mock_openai_client(answer="Generated answer"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_generation_sends_table_text_and_original_image(tmp_path, monkeypatch):
    from generation.rag_chain import generate_answer

    image_path = tmp_path / "chart.png"
    image_bytes = _png_bytes((20, 30, 40))
    image_path.write_bytes(image_bytes)
    client = _mock_openai_client()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", lambda: client)

    answer = generate_answer(
        "What trend is shown?",
        [
            {
                "uuid": "table-id",
                "type": "Table",
                "content": "<table>Revenue</table>",
                "metadata": {},
            },
            {
                "uuid": "image-id",
                "type": "Image",
                "content": "",
                "metadata": {"image_path": str(image_path)},
            },
        ],
    )

    assert answer == "Generated answer"
    request = client.chat.completions.create.call_args.kwargs
    user_content = request["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert "[Table]:\n<table>Revenue</table>" in user_content[0]["text"]
    assert "Question: What trend is shown?" in user_content[0]["text"]
    assert user_content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        },
    }


def test_generation_sends_multiple_labeled_images_in_context_order(
    tmp_path, monkeypatch
):
    from generation.rag_chain import generate_answer

    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    image_bytes = [_png_bytes((255, 0, 0)), _png_bytes((0, 0, 255))]
    for path, data in zip(image_paths, image_bytes):
        path.write_bytes(data)
    client = _mock_openai_client()
    monkeypatch.setattr("openai.OpenAI", lambda: client)

    generate_answer(
        "Compare these pages.",
        [
            {
                "uuid": "page-3",
                "type": "Image",
                "content": "",
                "metadata": {
                    "image_path": str(image_paths[0]),
                    "page": 3,
                    "document_id": "report",
                    "evidence_type": "full_page_render",
                    "evidence_label": "Document report, page 3",
                    "require_image_evidence": True,
                },
            },
            {
                "uuid": "page-1",
                "type": "Image",
                "content": "",
                "metadata": {
                    "image_path": str(image_paths[1]),
                    "page": 1,
                    "document_id": "report",
                    "evidence_type": "full_page_render",
                    "evidence_label": "Document report, page 1",
                    "require_image_evidence": True,
                },
            },
        ],
    )

    user_content = client.chat.completions.create.call_args.kwargs["messages"][1][
        "content"
    ]
    assert [block.get("text") for block in user_content[1::2]] == [
        "Image evidence: Document report, page 3",
        "Image evidence: Document report, page 1",
    ]
    assert [block["image_url"]["url"] for block in user_content[2::2]] == [
        "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        for data in image_bytes
    ]


def test_generation_uses_vision_summary_when_image_path_is_missing(
    tmp_path, monkeypatch
):
    from generation.rag_chain import generate_answer

    client = _mock_openai_client()
    monkeypatch.setattr("openai.OpenAI", lambda: client)

    generate_answer(
        "Describe this figure.",
        [
            {
                "uuid": "image-id",
                "type": "Image",
                "content": "",
                "vision_summary": "A rising revenue line.",
                "metadata": {"image_path": str(tmp_path / "missing.png")},
            }
        ],
    )

    user_content = client.chat.completions.create.call_args.kwargs["messages"][1][
        "content"
    ]
    assert len(user_content) == 1
    assert "[Figure]: A rising revenue line." in user_content[0]["text"]


def test_generation_rejects_missing_required_image_and_falls_back_for_corrupt_optional(
    tmp_path, monkeypatch, capsys
):
    from generation.rag_chain import generate_answer

    client = _mock_openai_client()
    monkeypatch.setattr("openai.OpenAI", lambda: client)
    missing = {
        "type": "Image",
        "content": "",
        "metadata": {
            "image_path": str(tmp_path / "missing.png"),
            "evidence_label": "Document report, page 2",
            "require_image_evidence": True,
        },
    }
    with pytest.raises(ValueError, match="Required image evidence"):
        generate_answer("What is on page 2?", [missing])

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")
    corrupt = {
        "type": "Image",
        "content": "",
        "vision_summary": "Stored chart description.",
        "metadata": {"image_path": str(corrupt_path)},
    }
    assert generate_answer("Describe this figure.", [corrupt]) == "Generated answer"
    user_content = client.chat.completions.create.call_args.kwargs["messages"][1][
        "content"
    ]
    assert "[Figure]: Stored chart description." in user_content[0]["text"]
    assert "is corrupt; using its text summary" in capsys.readouterr().err

    required_corrupt = {
        **corrupt,
        "metadata": {
            "image_path": str(corrupt_path),
            "evidence_label": "Visual match, page 2",
            "require_image_evidence": True,
        },
    }
    with pytest.raises(ValueError, match="corrupt or unreadable"):
        generate_answer("Describe this figure.", [required_corrupt])


def test_generation_does_not_swallow_openai_errors(monkeypatch):
    from generation.rag_chain import generate_answer

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    monkeypatch.setattr("openai.OpenAI", lambda: client)

    with pytest.raises(RuntimeError, match="API unavailable"):
        generate_answer("Question?", [{"type": "NarrativeText", "content": "Text"}])
