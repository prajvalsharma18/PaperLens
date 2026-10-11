from PIL import Image, ImageDraw

from ingestion.chart_extractor import extract_chart
from ingestion.data_extractor import extract_document_data
from evaluation.structured_extraction_benchmark import run_benchmark
from ingestion.table_extractor import extract_table
from retrieval.document_store import (
    load_structured_evidence,
    save_document_manifest,
    save_raw_element,
)
from retrieval.query_router import create_query_plan
from retrieval.structured_answers import answer_from_structured_data


def _table_record(document_id="report-1", element_id="table-1"):
    return extract_table(
        {
            "uuid": element_id,
            "type": "Table",
            "content": "",
            "metadata": {
                "document_id": document_id,
                "source": "report.pdf",
                "page": 2,
                "text_as_html": (
                    "<table><tr><th>Region</th><th>Revenue ($m)</th></tr>"
                    "<tr><td>North</td><td>12.5</td></tr>"
                    "<tr><td>South</td><td>18</td></tr></table>"
                ),
            },
        }
    )


def test_html_table_preserves_rows_headers_and_provenance():
    record = _table_record()

    assert record["status"] == "verified"
    assert record["data"]["headers"] == ["Region", "Revenue ($m)"]
    assert record["data"]["rows"][1:] == [
        ["North", "12.5"],
        ["South", "18"],
    ]
    assert record["source_element_id"] == "table-1"
    assert record["source_file"] == "report.pdf"
    assert record["page"] == 2


def test_plain_text_table_is_not_misreported_as_verified():
    record = extract_table(
        {
            "uuid": "plain-table",
            "type": "Table",
            "content": "Region\tRevenue\nNorth\t12\n",
            "metadata": {"document_id": "report-1"},
        }
    )

    assert record["status"] == "incomplete"
    assert record["data"]["rows"] == [["Region", "Revenue"], ["North", "12"]]
    assert record["limitations"]


def test_html_spans_keep_cell_positions_and_source_span_metadata():
    record = extract_table(
        {
            "uuid": "merged-table",
            "type": "Table",
            "metadata": {
                "document_id": "report-1",
                "text_as_html": (
                    "<table><tr><th rowspan='2'>Region</th>"
                    "<th colspan='2'>Sales</th></tr><tr><th>Q1</th><th>Q2</th></tr>"
                    "<tr><td>North</td><td>1</td><td>2</td></tr></table>"
                ),
            },
        }
    )

    assert record["data"]["rows"] == [
        ["Region", "Sales", ""],
        ["", "Q1", "Q2"],
        ["North", "1", "2"],
    ]
    assert record["data"]["cells"][0]["row_span"] == 2
    assert record["data"]["cells"][1]["column_span"] == 2


def test_malformed_html_is_marked_incomplete():
    record = extract_table(
        {
            "uuid": "malformed-table",
            "type": "Table",
            "metadata": {
                "document_id": "report-1",
                "text_as_html": "<table><tr><th>Metric</th><td>Value",
            },
        }
    )

    assert record["status"] == "incomplete"
    assert record["data"]["rows"] == [["Metric", "Value"]]
    assert record["warnings"]


def test_html_table_computes_extreme_and_comparison_from_verified_cells():
    record = _table_record()

    answer = answer_from_structured_data("Which region had the highest revenue?", [record])
    assert answer is not None
    assert "South: 18 Revenue ($m)" in answer
    assert "report.pdf, page 2, element table-1" in answer

    comparison = answer_from_structured_data(
        "What is the difference in revenue between North and South?", [record]
    )
    assert comparison is not None
    assert "absolute difference is 5.5" in comparison

    total = answer_from_structured_data(
        "What is the total revenue across all regions?", [record]
    )
    assert total is not None
    assert "North 12.5" in total
    assert "South 18" in total
    assert "= 30.5" in total


def test_incomplete_table_cannot_supply_numeric_answer():
    record = extract_table(
        {
            "uuid": "plain-table",
            "type": "Table",
            "content": "Region\tRevenue\nNorth\t12\n",
            "metadata": {"document_id": "report-1"},
        }
    )

    assert answer_from_structured_data("Which region had the highest revenue?", [record]) is None


def test_chart_digitization_is_estimated_and_cited(tmp_path):
    image_path = tmp_path / "synthetic-chart.png"
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 50, 45, 80), fill="black")
    draw.rectangle((70, 30, 85, 80), fill="black")
    image.save(image_path)

    element = {
        "uuid": "chart-1",
        "type": "Image",
        "metadata": {
            "document_id": "report-1",
            "source": "report.pdf",
            "page": 3,
            "image_path": str(image_path),
            "chart_type": "bar",
            "plot_area": [20, 10, 100, 90],
            "y_axis_label": "Revenue",
        },
    }
    ocr_result = {
        "tokens": [
            {"text": "10", "bbox": [4, 16, 14, 22]},
            {"text": "0", "bbox": [4, 78, 14, 84]},
            {"text": "North", "bbox": [27, 94, 48, 101]},
            {"text": "South", "bbox": [67, 94, 91, 101]},
        ]
    }

    record = extract_chart(
        element,
        document_id="report-1",
        ocr_reader=lambda _path, _page: ocr_result,
    )

    assert record["status"] == "estimated"
    assert [value["category"] for value in record["data"]["values"]] == [
        "North",
        "South",
    ]
    answer = answer_from_structured_data("Which bar has the highest revenue?", [record])
    assert answer is not None
    assert "estimated" in answer
    assert "report.pdf, page 3, element chart-1" in answer
    assert answer_from_structured_data("What is the exact value of the tallest bar?", [record])


def test_untyped_images_are_not_guessed_as_charts():
    record = extract_chart(
        {"uuid": "image-1", "type": "Image", "metadata": {"page": 1}},
        document_id="report-1",
    )

    assert record["status"] == "unsupported"
    assert record["warnings"]


def test_numeric_comparison_query_is_routed_for_structured_evidence():
    plan = create_query_plan("What is the difference between A and B in the chart?")
    assert plan.intent == "numeric_visual_question"
    assert plan.chart_digitization_desirable

    total_plan = create_query_plan("What is the total revenue across the chart?")
    assert total_plan.chart_digitization_desirable


def test_extract_document_data_persists_evidence_idempotently(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "store"))
    save_document_manifest("report-1", {"document_id": "report-1", "pages": []})
    save_raw_element(
        {
            "uuid": "table-1",
            "type": "Table",
            "content": "",
            "metadata": {
                "document_id": "report-1",
                "page": 2,
                "text_as_html": (
                    "<table><tr><th>Metric</th><th>Value</th></tr>"
                    "<tr><td>Sales</td><td>7</td></tr></table>"
                ),
            },
        }
    )

    first = extract_document_data("report-1")
    second = extract_document_data("report-1")

    assert first.status_counts == {"verified": 1}
    assert second.records[0]["record_id"] == first.records[0]["record_id"]
    assert load_structured_evidence("report-1") == first.records
    assert extract_document_data("report-1").failures == []


def test_synthetic_benchmark_reports_exact_matches_errors_and_provenance():
    scores = run_benchmark()

    assert scores["table_exact_match"] == 1.0
    assert scores["chart_category_association_accuracy"] == 1.0
    assert scores["estimated_value_pixel_mae"] is not None
    assert scores["estimated_value_pixel_mae"] < 0.1
    assert scores["direct_label_accuracy"] is None
    assert scores["provenance_correct"] == scores["provenance_denominator"] == 4
