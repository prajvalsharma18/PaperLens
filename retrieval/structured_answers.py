"""Deterministic, provenance-cited comparisons over extracted numeric evidence."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "bar",
    "bars",
    "chart",
    "compare",
    "exact",
    "for",
    "has",
    "highest",
    "how",
    "in",
    "is",
    "lowest",
    "maximum",
    "minimum",
    "of",
    "on",
    "the",
    "tallest",
    "value",
    "values",
    "what",
    "which",
}


def _number(value: Any) -> tuple[Decimal, str] | None:
    text = str(value).strip()
    matches = list(_NUMBER.finditer(text))
    if len(matches) != 1:
        return None
    raw = matches[0].group(0)
    unit = "%" if raw.endswith("%") else ""
    try:
        return Decimal(raw.rstrip("%").replace(",", "")), unit
    except InvalidOperation:
        return None


def _question_direction(question: str) -> str | None:
    lowered = question.lower()
    if re.search(r"\b(highest|highest|maximum|max|largest|tallest|most)\b", lowered):
        return "max"
    if re.search(r"\b(lowest|minimum|min|smallest|shortest|least)\b", lowered):
        return "min"
    return None


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS
    }


def _provenance(record: dict[str, Any]) -> str:
    source = record.get("source_file") or f"document {record['document_id']}"
    page = record.get("page")
    page_text = f", page {page}" if page is not None else ""
    return (
        f"{source}{page_text}, element {record['source_element_id']} "
        f"[document ID: {record['document_id']}]"
    )


def _chart_observations(
    question: str,
    record: dict[str, Any],
) -> list[tuple[str, Decimal, str]]:
    data = record.get("data", {})
    scope = " ".join(
        str(data.get(key) or "") for key in ("title", "x_axis_label", "y_axis_label")
    )
    query_terms = _tokens(question)
    scope_terms = _tokens(scope)
    labels = data.get("values", [])
    label_terms = set().union(
        *(
            _tokens(str(item.get("category") or item.get("series") or ""))
            for item in labels
            if isinstance(item, dict)
        )
    ) if labels else set()
    if query_terms and scope_terms and not query_terms.intersection(scope_terms | label_terms):
        return []
    values: list[tuple[str, Decimal, str]] = []
    for item in data.get("values", []):
        parsed = _number(item.get("value"))
        if parsed is not None:
            label = str(item.get("category") or item.get("series") or "")
            values.append((label, parsed[0], parsed[1] or str(item.get("unit") or "")))
    return values


def _table_observations(
    question: str,
    record: dict[str, Any],
) -> tuple[str, list[tuple[str, Decimal, str]]]:
    if record.get("status") != "verified":
        return "", []
    data = record.get("data", {})
    rows = data.get("rows", [])
    headers = data.get("headers", [])
    if not rows or not headers:
        return "", []
    query_terms = _tokens(question)
    header_rows = data.get("header_rows") or [0]
    relevant_columns = [
        index
        for index, header in enumerate(headers)
        if query_terms.intersection(_tokens(str(header)))
        and any(
            index < len(row) and _number(row[index]) is not None
            for row_number, row in enumerate(rows)
            if row_number > header_rows[0]
        )
    ]
    if len(relevant_columns) != 1:
        return "", []
    column = relevant_columns[0]
    header_row = header_rows[0]
    values: list[tuple[str, Decimal, str]] = []
    for row_index, row in enumerate(rows):
        if row_index <= header_row or column >= len(row):
            continue
        parsed = _number(row[column])
        if parsed is None:
            continue
        label = next(
            (
                str(value).strip()
                for index, value in enumerate(row)
                if index != column and str(value).strip()
            ),
            f"row {row_index + 1}",
        )
        values.append((label, parsed[0], parsed[1]))
    return str(headers[column]), values


def _answer_comparison(
    question: str,
    record: dict[str, Any],
    observations: list[tuple[str, Decimal, str]],
) -> str | None:
    if not re.search(r"\b(compare|difference|delta|versus|vs\.?)\b", question, re.I):
        return None
    query_terms = _tokens(question)
    selected = [
        item
        for item in observations
        if item[0] and _tokens(item[0]).intersection(query_terms)
    ]
    if len(selected) != 2 or selected[0][2] != selected[1][2]:
        return None
    first, second = selected
    difference = abs(first[1] - second[1])
    unit = first[2]
    difference_unit = " percentage points" if unit == "%" else unit
    qualifier = ""
    if record["kind"] == "chart":
        qualifier = (
            "estimated "
            if record.get("status") == "estimated"
            or record.get("extraction_method") == "axis_digitization"
            else "OCR-transcribed "
        )
    return (
        f"{first[0]} is {qualifier}{first[1].normalize():f}{unit} and "
        f"{second[0]} is {qualifier}{second[1].normalize():f}{unit}; "
        f"the absolute difference is {difference.normalize():f}{difference_unit}. "
        f"Source: {_provenance(record)}. "
        f"Extraction: {record['extraction_method']}."
    )


def _answer_sum(
    question: str,
    record: dict[str, Any],
    observations: list[tuple[str, Decimal, str]],
) -> str | None:
    if not re.search(r"\b(sum|total)\b", question, re.I) or not observations:
        return None
    query_terms = _tokens(question)
    if re.search(r"\b(all|across|overall|values|rows|bars)\b", question, re.I):
        selected = observations
    else:
        selected = [
            item
            for item in observations
            if item[0] and _tokens(item[0]).intersection(query_terms)
        ]
    if len(selected) < 2 or len({item[2] for item in selected}) != 1:
        return None
    total = sum((item[1] for item in selected), Decimal(0))
    qualifier = ""
    caveat = ""
    if record["kind"] == "chart":
        if record.get("status") == "estimated" or record.get(
            "extraction_method"
        ) == "axis_digitization":
            qualifier = "estimated "
            caveat = " The chart-derived values and total are estimates."
        else:
            qualifier = "OCR-transcribed "
            caveat = " OCR labels and their associations are not verified."
    terms = " + ".join(
        f"{item[0]} {qualifier}{item[1].normalize():f}{item[2]}"
        for item in selected
    )
    return (
        f"The total is {terms} = {total.normalize():f}{selected[0][2]}. "
        f"Source: {_provenance(record)}. "
        f"Extraction: {record['extraction_method']}.{caveat}"
    )


def _answer_extreme(
    record: dict[str, Any],
    direction: str,
    observations: list[tuple[str, Decimal, str]],
    table_header: str = "",
) -> str | None:
    if not observations:
        return None
    target = (max if direction == "max" else min)(value[1] for value in observations)
    winners = [entry for entry in observations if entry[1] == target]
    descriptions = []
    for label, number, unit in winners:
        qualifier = ""
        if record["kind"] == "chart":
            qualifier = (
                "estimated "
                if record.get("status") == "estimated"
                or record.get("extraction_method") == "axis_digitization"
                else "OCR-transcribed "
            )
        formatted = f"{number.normalize():f}{unit}"
        suffix = f" {table_header}" if table_header else ""
        descriptions.append(f"{label or 'Value'}: {qualifier}{formatted}{suffix}")
    action = "highest" if direction == "max" else "lowest"
    caveat = ""
    if record["kind"] == "chart":
        if record.get("status") == "estimated":
            caveat = " These values are estimates, not exact source data."
        elif record.get("status") != "verified":
            caveat = " OCR transcriptions and their associations are not verified."
    return (
        f"The {action} recovered value is "
        + " and ".join(descriptions)
        + f". Source: {_provenance(record)}. "
        + f"Extraction: {record['extraction_method']}.{caveat}"
    )


def _answer_lookup(
    question: str,
    record: dict[str, Any],
    observations: list[tuple[str, Decimal, str]],
) -> str | None:
    query_terms = _tokens(question)
    selected = [
        item
        for item in observations
        if item[0] and _tokens(item[0]).intersection(query_terms)
    ]
    if len(selected) != 1:
        return None
    label, number, unit = selected[0]
    qualifier = ""
    caveat = ""
    if record["kind"] == "chart":
        if record.get("status") == "estimated" or record.get(
            "extraction_method"
        ) == "axis_digitization":
            qualifier = "estimated "
            caveat = " This is not the exact underlying source value."
        else:
            qualifier = "OCR-transcribed "
            caveat = " The OCR transcription and association are not verified."
    return (
        f"{label}: {qualifier}{number.normalize():f}{unit}. "
        f"Source: {_provenance(record)}. "
        f"Extraction: {record['extraction_method']}.{caveat}"
    )


def answer_from_structured_data(
    question: str,
    records: list[dict[str, Any]],
) -> str | None:
    """Answer clear extrema or two-item comparisons when evidence is unambiguous."""
    direction = _question_direction(question)
    is_lookup = bool(re.search(r"\b(value|amount|number|how many|what was)\b", question, re.I))
    is_comparison = bool(
        re.search(
            r"\b(compare|difference|delta|versus|vs\.?|sum|total)\b",
            question,
            re.I,
        )
    )
    is_sum = bool(re.search(r"\b(sum|total)\b", question, re.I))
    if direction is None and not is_comparison and not is_sum and not is_lookup:
        return None
    answers = []
    for record in records:
        if record.get("kind") == "chart":
            observations = _chart_observations(question, record)
            table_header = ""
        elif record.get("kind") == "table":
            table_header, observations = _table_observations(question, record)
        else:
            continue
        answer = _answer_sum(question, record, observations)
        if answer is None:
            answer = _answer_comparison(question, record, observations)
        if answer is None and is_lookup:
            answer = _answer_lookup(question, record, observations)
        if answer is None and direction is not None:
            answer = _answer_extreme(
                record,
                direction,
                observations,
                table_header,
            )
        if answer:
            answers.append(answer)
    return answers[0] if len(answers) == 1 else None
