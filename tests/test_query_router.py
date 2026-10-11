import pytest

from retrieval.query_router import (
    QueryRoutingError,
    create_query_plan,
    extract_page_references,
)


@pytest.mark.parametrize(
    ("question", "intent", "pages", "strategy"),
    [
        ("Summarize the report.", "general_question", (), "semantic_search"),
        (
            "Compare the architectural differences between page 1 and page 2.",
            "page_comparison",
            (1, 2),
            "exact_page",
        ),
        (
            "Compare layouts on pages 1 and 3.",
            "page_comparison",
            (1, 3),
            "exact_page",
        ),
        (
            "Compare pages 1, 2, and 3.",
            "page_comparison",
            (1, 2, 3),
            "exact_page",
        ),
        ("What is shown on page 4?", "page_question", (4,), "exact_page"),
        (
            "Explain the chart on page 6.",
            "page_question",
            (6,),
            "exact_page",
        ),
        (
            "Find the figure showing the system architecture.",
            "visual_question",
            (),
            "semantic_visual_search",
        ),
        (
            "What is the exact value of the tallest bar?",
            "numeric_visual_question",
            (),
            "semantic_visual_search",
        ),
    ],
)
def test_query_plan_routes_supported_intents(question, intent, pages, strategy):
    plan = create_query_plan(question, document_id="selected-document")

    assert plan.intent == intent
    assert plan.page_numbers == pages
    assert plan.retrieval_strategy == strategy
    assert plan.document_id == "selected-document"


def test_explicit_page_ranges_and_visual_references_are_normalized():
    plan = create_query_plan(
        "Compare pages 2-4 and figure 7 with table no. 3.",
        document_id="report",
    )

    assert plan.page_numbers == (2, 3, 4)
    assert plan.visual_references == ("figure 7", "table 3")
    assert plan.requires_full_page_visual
    assert plan.requires_multiple_evidence


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What happened in 2024?", ()),
        ("What is shown on page 1?", (1,)),
        ("Compare pages 4-2.", ()),
    ],
)
def test_router_does_not_treat_arbitrary_numbers_as_pages(question, expected):
    if question == "Compare pages 4-2.":
        with pytest.raises(QueryRoutingError, match="descending page range"):
            extract_page_references(question)
    else:
        assert extract_page_references(question) == expected


def test_router_rejects_page_zero():
    with pytest.raises(QueryRoutingError, match="Page numbers start at 1"):
        extract_page_references("What is shown on page 0?")


def test_router_rejects_unreasonably_large_page_ranges():
    with pytest.raises(QueryRoutingError, match="at most 100 pages"):
        create_query_plan("Summarize pages 1-101.")
