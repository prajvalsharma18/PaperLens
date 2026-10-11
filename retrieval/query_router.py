"""Deterministic query intent and retrieval-strategy planning."""

from __future__ import annotations

import re
from dataclasses import dataclass


class QueryRoutingError(ValueError):
    """Raised when a query contains an invalid explicit page reference."""


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    document_id: str | None
    page_numbers: tuple[int, ...]
    visual_references: tuple[str, ...]
    visual_question: bool
    requires_full_page_visual: bool
    requires_multiple_evidence: bool
    chart_digitization_desirable: bool
    retrieval_strategy: str


_PAGE_REFERENCE = re.compile(
    r"\bpages?\s+(?:(?:no\.?|number)\s*)?"
    r"(\d+(?:(?:\s*,\s*(?:and\s*)?|\s+(?:and|to|&)\s*|\s*-\s*)\d+)*)",
    re.IGNORECASE,
)
_PAGE_TOKEN = re.compile(r"\d+|,|and|to|&|-", re.IGNORECASE)
_VISUAL_REFERENCE = re.compile(
    r"\b(figure|fig\.?|table|chart|diagram|image)\s*(?:no\.?\s*)?(\d+)\b",
    re.IGNORECASE,
)
_VISUAL_TERMS = re.compile(
    r"\b(figure|figures|chart|charts|table|tables|diagram|diagrams|"
    r"image|images|graph|graphs|plot|plots|layout|layouts|visual|visuals|bar|bars)\b",
    re.IGNORECASE,
)
_NUMERIC_VISUAL_TERMS = re.compile(
    r"\b(exact|value|values|number|amount|how many|tallest|highest|"
    r"lowest|shortest|numeric|numerical|percentage|percent|compare|comparison|"
    r"difference|delta|calculation|calculate|computed|sum|total)\b",
    re.IGNORECASE,
)
_COMPARISON_TERMS = re.compile(
    r"\b(compare|comparison|contr(ast|asts)|versus|vs\.?|differences?\s+between)\b",
    re.IGNORECASE,
)


def _expand_page_expression(expression: str) -> list[int]:
    tokens = [match.group(0).lower() for match in _PAGE_TOKEN.finditer(expression)]
    pages: list[int] = []
    index = 0
    while index < len(tokens):
        if not tokens[index].isdigit():
            index += 1
            continue
        start = int(tokens[index])
        pages.append(start)
        index += 1
        while index + 1 < len(tokens) and tokens[index] in {
            ",",
            "and",
            "&",
            "to",
            "-",
        } and tokens[index + 1].isdigit():
            separator = tokens[index]
            end = int(tokens[index + 1])
            if separator in {"to", "-"}:
                if end < start:
                    raise QueryRoutingError(
                        f"Invalid descending page range {start}-{end}."
                    )
                if end - start + 1 > 100:
                    raise QueryRoutingError(
                        "A page range may contain at most 100 pages."
                    )
                pages.extend(range(start + 1, end + 1))
            else:
                pages.append(end)
            start = end
            index += 2
    return pages


def extract_page_references(question: str) -> tuple[int, ...]:
    """Extract only numbers explicitly introduced as page references."""
    pages: list[int] = []
    for match in _PAGE_REFERENCE.finditer(question):
        pages.extend(_expand_page_expression(match.group(1)))
    if any(page < 1 for page in pages):
        raise QueryRoutingError("Page numbers start at 1.")
    return tuple(pages)


def create_query_plan(
    question: str,
    document_id: str | None = None,
) -> QueryPlan:
    """Build a query plan without external model calls."""
    page_numbers = extract_page_references(question)
    comparison = bool(_COMPARISON_TERMS.search(question))
    visual = bool(_VISUAL_TERMS.search(question))
    numeric_visual = visual and bool(_NUMERIC_VISUAL_TERMS.search(question))
    references = tuple(
        f"{match.group(1).lower().rstrip('.')} {match.group(2)}"
        for match in _VISUAL_REFERENCE.finditer(question)
    )

    if page_numbers and comparison and len(page_numbers) > 1:
        intent = "page_comparison"
    elif page_numbers:
        intent = "page_question"
    elif numeric_visual:
        intent = "numeric_visual_question"
    elif visual:
        intent = "visual_question"
    else:
        intent = "general_question"

    if page_numbers:
        strategy = "exact_page"
    elif visual:
        strategy = "semantic_visual_search"
    else:
        strategy = "semantic_search"

    return QueryPlan(
        intent=intent,
        document_id=document_id,
        page_numbers=page_numbers,
        visual_references=references,
        visual_question=visual,
        requires_full_page_visual=bool(page_numbers),
        requires_multiple_evidence=len(page_numbers) > 1,
        chart_digitization_desirable=numeric_visual,
        retrieval_strategy=strategy,
    )
