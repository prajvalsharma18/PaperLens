"""Tests for multimodal prompt construction without OpenAI requests."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _mock_openai_client(answer="Generated answer"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_generation_sends_table_text_and_original_image(tmp_path, monkeypatch):
    from generation.rag_chain import generate_answer

    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"chart-image-bytes")
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
            + base64.b64encode(b"chart-image-bytes").decode("ascii")
        },
    }


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


def test_generation_does_not_swallow_openai_errors(monkeypatch):
    from generation.rag_chain import generate_answer

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    monkeypatch.setattr("openai.OpenAI", lambda: client)

    with pytest.raises(RuntimeError, match="API unavailable"):
        generate_answer("Question?", [{"type": "NarrativeText", "content": "Text"}])
