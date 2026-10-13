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
            "structured_numeric_search",
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
    "question",
    [
        "Search across all indexed documents for a diagram or flowchart of a process. Identify the matching document and page number.",
        "Which document and source PDF contains the visual explanation? Identify the page.",
        "Find the diagram and report its page number.",
        "Find a diagram and identify its source PDF and page number.",
    ],
)
def test_page_provenance_language_does_not_request_numeric_chart_analysis(question):
    plan = create_query_plan(question)

    assert plan.intent == "visual_question"
    assert plan.visual_question
    assert not plan.chart_digitization_desirable
    assert plan.retrieval_strategy == "semantic_visual_search"


@pytest.mark.parametrize(
    "question",
    [
        "Find the photo with a modern facade and large glass windows.",
        "Find a photograph showing the modern glass facade.",
        "Search for a picture of the building with large windows.",
        "Locate the image that shows the modern facade.",
        "Find visual evidence of the building's glass windows.",
        "Find the modern facade with large glass windows.",
    ],
)
def test_natural_visual_search_phrases_use_semantic_visual_retrieval(question):
    plan = create_query_plan(question)

    assert plan.visual_question
    assert plan.retrieval_strategy == "semantic_visual_search"
    assert not plan.chart_digitization_desirable


@pytest.mark.parametrize(
    ("question", "intent", "strategy", "chart_digitization"),
    [
        ("What does the paper say about image processing?", "general_question", "semantic_search", False),
        ("Summarize the author's conclusion about the experiment.", "general_question", "semantic_search", False),
        ("Find the photo with large windows on page 2.", "page_question", "exact_page", False),
        ("What is the exact value of the tallest bar?", "numeric_visual_question", "structured_numeric_search", True),
        ("What is the maximum value in the chart?", "numeric_visual_question", "structured_numeric_search", True),
        ("What is the difference between A and B in the chart?", "numeric_visual_question", "structured_numeric_search", True),
        ("Calculate the total of the bars in the chart.", "numeric_visual_question", "structured_numeric_search", True),
        ("How many windows are visible in the photo?", "visual_question", "semantic_visual_search", False),
    ],
)
def test_visual_routing_preserves_text_page_and_numeric_intents(
    question, intent, strategy, chart_digitization
):
    plan = create_query_plan(question)

    assert plan.intent == intent
    assert plan.retrieval_strategy == strategy
    assert plan.chart_digitization_desirable is chart_digitization


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
