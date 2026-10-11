"""Small deterministic checks for controlled answer evaluation fixtures."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


def evaluate_answer_case(answer: str, expected: dict[str, Any]) -> dict[str, Any]:
    lowered = answer.casefold()
    required_facts = expected.get("required_facts", [])
    forbidden_facts = expected.get("forbidden_facts", [])
    citations = expected.get("required_citations", [])
    available_ids = set(expected.get("available_evidence_ids", []))
    cited_ids = set(re.findall(r"\[(E-[0-9a-f]{10})\]", answer))
    invalid_citations = sorted(cited_ids - available_ids) if "available_evidence_ids" in expected else []
    fact_results = {fact: str(fact).casefold() in lowered for fact in required_facts}
    forbidden_results = {fact: str(fact).casefold() not in lowered for fact in forbidden_facts}
    citation_results = {citation: str(citation) in answer for citation in citations}
    numeric = expected.get("numeric")
    numeric_correct: bool | None = None
    if numeric is not None:
        try:
            target = Decimal(str(numeric["value"]))
            tolerance = Decimal(str(numeric.get("tolerance", "0")))
            actuals = [Decimal(value.replace(",", "")) for value in re.findall(
                r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?",
                answer,
            )]
            numeric_correct = any(abs(value - target) <= tolerance for value in actuals)
            unit = numeric.get("unit")
            if unit:
                numeric_correct = numeric_correct and str(unit).casefold() in lowered
        except (KeyError, InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid expected numeric answer: {exc}") from exc
    abstention_ok: bool | None = None
    if "should_abstain" in expected:
        signals = ("cannot determine", "insufficient evidence", "cannot provide", "unable to determine")
        detected = any(signal in lowered for signal in signals)
        abstention_ok = detected is bool(expected["should_abstain"])
    return {
        "required_facts": fact_results,
        "forbidden_facts": forbidden_results,
        "required_citations": citation_results,
        "invalid_citations": invalid_citations,
        "numeric_correct": numeric_correct,
        "abstention_correct": abstention_ok,
        "passed": (
            all(fact_results.values())
            and all(forbidden_results.values())
            and all(citation_results.values())
            and not invalid_citations
            and numeric_correct is not False
            and abstention_ok is not False
        ),
    }
