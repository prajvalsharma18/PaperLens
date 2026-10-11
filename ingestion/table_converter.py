"""
Converts raw Table elements to HTML or Markdown for LLM-friendly embedding.
"""

from typing import Any


def convert_tables(table_elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert table elements to HTML/Markdown representation.

    Unstructured provides table HTML in element.metadata.text_as_html
    when infer_table_structure=True. This function normalises that
    into the document format used by the rest of the pipeline.
    """
    converted = []
    for el in table_elements:
        html = el.get("metadata", {}).get("text_as_html") or el.get("content", "")
        converted.append({
            **el,
            "type": "Table",
            "content": html,
            "embed_content": f"Table content:\n{html}",  # What gets embedded
        })
    return converted
