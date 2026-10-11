"""Deterministic parsing of preserved table HTML or delimited plain text."""

from __future__ import annotations

import csv
import io
import re
from html.parser import HTMLParser
from typing import Any

from ingestion.structured_evidence import make_structured_record


def _normalize_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._cell_text: list[str] = []
        self._table_depth = 0
        self._nested_depth = 0
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            if self._table_depth:
                self._nested_depth += 1
            else:
                self._table_depth = 1
            return
        if not self._table_depth or self._nested_depth:
            return
        if tag == "tr":
            if self._row is not None:
                self.rows.append(self._row)
                self.malformed = True
            self._row = []
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
                self.malformed = True
            if self._cell is not None:
                self._finish_cell()
                self.malformed = True
            self._cell = {
                "header": tag == "th",
                "row_span": _positive_span(attributes.get("rowspan")),
                "column_span": _positive_span(attributes.get("colspan")),
            }
            self._cell_text = []
        elif tag == "br" and self._cell is not None:
            self._cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._nested_depth:
            self._nested_depth -= 1
            return
        if not self._table_depth or self._nested_depth:
            return
        if tag in {"td", "th"}:
            if self._cell is None:
                self.malformed = True
            else:
                self._finish_cell()
        elif tag == "tr":
            if self._cell is not None:
                self._finish_cell()
                self.malformed = True
            if self._row is not None:
                self.rows.append(self._row)
                self._row = None
            else:
                self.malformed = True
        elif tag == "table":
            if self._cell is not None:
                self._finish_cell()
                self.malformed = True
            if self._row is not None:
                self.rows.append(self._row)
                self._row = None
                self.malformed = True
            self._table_depth = 0

    def handle_data(self, data: str) -> None:
        if self._cell is not None and self._nested_depth == 0:
            self._cell_text.append(data)

    def close(self) -> None:
        super().close()
        if self._cell is not None:
            self._finish_cell()
            self.malformed = True
        if self._row is not None:
            self.rows.append(self._row)
            self._row = None
            self.malformed = True
        if self._table_depth:
            self.malformed = True

    def _finish_cell(self) -> None:
        assert self._cell is not None
        assert self._row is not None
        self._cell["value"] = _normalize_cell("".join(self._cell_text))
        self._row.append(self._cell)
        self._cell = None
        self._cell_text = []


def _positive_span(value: str | None) -> int:
    try:
        return max(1, int(value or "1"))
    except ValueError:
        return 1


def _expand_html_rows(
    parsed_rows: list[list[dict[str, Any]]],
) -> tuple[list[list[str]], list[dict[str, Any]], list[int]]:
    cell_records: list[dict[str, Any]] = []
    header_rows: list[int] = []
    grid: dict[tuple[int, int], str] = {}
    max_row = len(parsed_rows) - 1
    max_column = -1
    for row_index, cells in enumerate(parsed_rows):
        column = 0
        row_is_header = False
        for cell in cells:
            while (row_index, column) in grid:
                column += 1
            value = cell["value"]
            row_is_header = row_is_header or cell["header"]
            cell_records.append(
                {
                    "row": row_index,
                    "column": column,
                    "value": value,
                    "header": cell["header"],
                    "row_span": cell["row_span"],
                    "column_span": cell["column_span"],
                }
            )
            for row_offset in range(cell["row_span"]):
                for column_offset in range(cell["column_span"]):
                    grid[(row_index + row_offset, column + column_offset)] = (
                        value if row_offset == 0 and column_offset == 0 else ""
                    )
            max_row = max(max_row, row_index + cell["row_span"] - 1)
            max_column = max(
                max_column,
                column + cell["column_span"] - 1,
            )
            column += cell["column_span"]
        if row_is_header:
            header_rows.append(row_index)

    rows = [
        [grid.get((row_index, column), "") for column in range(max_column + 1)]
        for row_index in range(max_row + 1)
    ]
    return rows, cell_records, header_rows


def _plain_text_rows(text: str) -> list[list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if any("\t" in line for line in lines):
        rows = [[_normalize_cell(value) for value in line.split("\t")] for line in lines]
    elif all("|" in line for line in lines):
        rows = [
            [_normalize_cell(value) for value in line.strip("|").split("|")]
            for line in lines
        ]
        rows = [
            row
            for row in rows
            if not all(re.fullmatch(r":?-{3,}:?", value or "") for value in row)
        ]
    else:
        rows = [[_normalize_cell(line)] for line in lines]
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def extract_table(
    element: dict[str, Any],
    *,
    document_id: str | None = None,
) -> dict[str, Any]:
    metadata = element.get("metadata", {})
    identity = str(element.get("uuid") or metadata.get("element_id") or "")
    owner = document_id or metadata.get("document_id")
    if not identity or not owner:
        raise ValueError("Table extraction requires document_id and a stable element ID.")

    source_html = metadata.get("text_as_html")
    source = str(source_html if source_html else element.get("content", ""))
    warnings: list[str] = []
    if source_html or re.search(r"<\s*table\b", source, re.IGNORECASE):
        parser = _TableParser()
        try:
            parser.feed(source)
            parser.close()
            rows, cells, header_rows = _expand_html_rows(parser.rows)
        except (AssertionError, ValueError) as exc:
            rows, cells, header_rows = [], [], []
            parser.malformed = True
            warnings.append(f"Table markup could not be fully parsed: {exc}")
        if parser.malformed:
            warnings.append("Malformed table markup was repaired where possible.")
        headers = rows[header_rows[0]] if header_rows else []
        repeated_headers = [
            index
            for index, row in enumerate(rows)
            if headers and index > header_rows[0] and row == headers
        ]
        status = "incomplete" if parser.malformed else "verified"
        method = "html_table_parser"
        data = {
            "rows": rows,
            "headers": headers,
            "header_rows": header_rows,
            "repeated_header_rows": repeated_headers,
            "cells": cells,
            "source_representation": "html",
            "source_representation_text": source,
        }
        if not rows:
            status = "unreadable"
            warnings.append("No table rows could be recovered from the HTML.")
    else:
        rows = _plain_text_rows(source)
        status = "incomplete" if rows else "unreadable"
        method = "plain_text_table_fallback"
        warnings.append(
            "Plain text does not preserve reliable table cell boundaries or merged-cell structure."
        )
        data = {
            "rows": rows,
            "headers": rows[0] if len(rows) > 1 else [],
            "header_rows": [0] if len(rows) > 1 else [],
            "repeated_header_rows": [],
            "cells": [],
            "source_representation": "plain_text",
            "source_representation_text": source,
        }
    return make_structured_record(
        kind="table",
        document_id=str(owner),
        source_element_id=identity,
        source_file=metadata.get("source_file", metadata.get("source")),
        page=metadata.get("page"),
        extraction_method=method,
        status=status,
        source_image_path=metadata.get("image_path"),
        source_bbox=metadata.get("bbox"),
        data=data,
        warnings=warnings,
        limitations=(
            ["A plain-text fallback cannot claim that full table structure was recovered."]
            if method == "plain_text_table_fallback"
            else []
        ),
    )


def extract_table_image(
    element: dict[str, Any],
    *,
    document_id: str,
    ocr_result: dict[str, Any] | None = None,
    ocr_error: str | None = None,
) -> dict[str, Any]:
    metadata = element.get("metadata", {})
    identity = str(element.get("uuid") or metadata.get("element_id") or "")
    if not identity:
        raise ValueError("Image table extraction requires a stable element ID.")
    warnings = []
    if ocr_result:
        recognized_text = str(ocr_result.get("text", "")).strip()
        tokens = ocr_result.get("tokens", [])
        warnings.append(
            "OCR text is not cell segmentation; table boundaries and merged cells are unverified."
        )
        data = {
            "rows": [],
            "headers": [],
            "header_rows": [],
            "repeated_header_rows": [],
            "cells": [],
            "source_representation": "image_ocr",
            "recognized_text": recognized_text,
            "ocr_tokens": tokens,
        }
        status = "incomplete" if recognized_text else "unreadable"
    else:
        data = {
            "rows": [],
            "headers": [],
            "header_rows": [],
            "repeated_header_rows": [],
            "cells": [],
            "source_representation": "image",
            "recognized_text": "",
            "ocr_tokens": [],
        }
        status = "incomplete"
        warnings.append(ocr_error or "No OCR result is available for this image table.")
    return make_structured_record(
        kind="table",
        document_id=document_id,
        source_element_id=identity,
        source_file=metadata.get("source_file", metadata.get("source")),
        page=metadata.get("page"),
        source_image_path=metadata.get("image_path"),
        source_bbox=metadata.get("bbox"),
        extraction_method="tesseract_ocr",
        status=status,
        data=data,
        warnings=warnings,
        limitations=["OCR does not recover cell boundaries or merged-cell structure."],
    )


def table_to_csv(record: dict[str, Any]) -> str:
    if record.get("kind") != "table":
        raise ValueError("CSV serialization requires a table record.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(record.get("data", {}).get("rows", []))
    return output.getvalue()


def table_to_json(record: dict[str, Any]) -> str:
    if record.get("kind") != "table":
        raise ValueError("JSON serialization requires a table record.")
    import json

    return json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
