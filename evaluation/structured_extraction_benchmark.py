"""Small deterministic benchmark for table/chart extraction mechanics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ingestion.chart_extractor import extract_chart
from ingestion.table_extractor import extract_table


_FIXTURE_PATH = Path(__file__).parent / "datasets" / "synthetic-extraction-fixtures.json"


def run_benchmark() -> dict[str, Any]:
    fixture_dataset = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_cases = {
        case["extraction_evaluation"]["source_element_id"]: case["extraction_evaluation"]
        for case in fixture_dataset["cases"]
    }
    table = extract_table(
        {
            "uuid": "benchmark-table",
            "type": "Table",
            "metadata": {
                "document_id": "synthetic-benchmark",
                "source": "synthetic.html",
                "page": 1,
                "text_as_html": (
                    "<table><tr><th>Category</th><th>Value</th></tr>"
                    "<tr><td>A</td><td>10</td></tr>"
                    "<tr><td>B</td><td>20</td></tr></table>"
                ),
            },
        }
    )
    table_expected = fixture_cases["benchmark-table"]
    expected_rows = table_expected["expected_rows"]
    incomplete_table = extract_table(
        {
            "uuid": "benchmark-incomplete-table",
            "type": "Table",
            "content": "Region\tRevenue\nNorth\t12",
            "metadata": {"document_id": "synthetic-benchmark", "page": 3},
        }
    )

    with tempfile.TemporaryDirectory(prefix="paperlens-benchmark-") as temp_dir:
        image_path = Path(temp_dir) / "synthetic-chart.png"
        image = Image.new("RGB", (120, 120), "white")
        try:
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 50, 45, 80), fill="black")
            draw.rectangle((70, 30, 85, 80), fill="black")
            image.save(image_path)
        finally:
            image.close()
        chart_element = {
            "uuid": "benchmark-chart",
            "type": "Image",
            "metadata": {
                "document_id": "synthetic-benchmark",
                "source": "synthetic-chart.png",
                "page": 2,
                "image_path": str(image_path),
                "chart_type": "bar",
                "plot_area": [20, 10, 100, 90],
                "y_axis_label": "Value",
            },
        }
        tokens: list[dict[str, Any]] = [
            {"text": "10", "bbox": [4, 16, 14, 24]},
            {"text": "0", "bbox": [4, 76, 14, 84]},
            {"text": "A", "bbox": [34, 94, 42, 102]},
            {"text": "B", "bbox": [74, 94, 82, 102]},
        ]
        chart = extract_chart(
            chart_element,
            document_id="synthetic-benchmark",
            ocr_reader=lambda *args: {
                "tokens": tokens if len(args) == 2 else []
            },
        )

    table_rows = table["data"]["rows"]
    table_exact_match = float(table_rows == expected_rows)
    expected_cell_count = sum(map(len, expected_rows))
    actual_cell_count = sum(map(len, table_rows))
    matching_cells = sum(
        actual == expected
        for actual_row, expected_row in zip(table_rows, expected_rows)
        for actual, expected in zip(actual_row, expected_row)
    )
    chart_values = chart.get("data", {}).get("values", [])
    category_matches = sum(
        actual.get("category") == category
        for actual, category in zip(chart_values, ("A", "B"))
    )
    category_association_accuracy = (
        category_matches / 2 if len(chart_values) == 2 else 0.0
    )
    chart_expected = fixture_cases["benchmark-chart"]
    expected = chart_expected["expected_values"]
    errors = [
        abs(float(item["value"]) - expected[item["category"]])
        for item in chart_values
        if item.get("category") in expected and item.get("value_source") == "axis_digitization"
    ]
    pixel_value_mae = sum(errors) / len(errors) if errors else None
    unsupported_chart = extract_chart(
        {
            "uuid": "benchmark-unsupported-chart",
            "type": "Image",
            "metadata": {"chart_type": "line", "page": 4},
        },
        document_id="synthetic-benchmark",
    )
    provenance_records = [table, incomplete_table, chart, unsupported_chart]
    expected_provenance = {
        "benchmark-table": 1,
        "benchmark-incomplete-table": 3,
        "benchmark-chart": 2,
        "benchmark-unsupported-chart": 4,
    }
    provenance_correct = sum(
        record.get("source_element_id") == element_id
        and record.get("page") == expected_page
        for record, (element_id, expected_page) in zip(
            provenance_records, expected_provenance.items()
        )
    )
    mae = sum(errors) / len(errors) if errors else None
    tolerance = chart_expected["numeric_tolerance"]
    expected_statuses = tuple(
        fixture_cases[record["source_element_id"]]["expected_status"]
        for record in provenance_records
    )
    status_correct = sum(
        record.get("status") == expected
        for record, expected in zip(provenance_records, expected_statuses)
    )
    return {
        "evaluation_kind": "synthetic_fixture_execution",
        "dataset_id": fixture_dataset["dataset_id"],
        "fixture_case_count": len(fixture_dataset["cases"]),
        "table_exact_match": table_exact_match,
        "table_cell_exact_match": matching_cells / expected_cell_count,
        "table_cells_correct": matching_cells,
        "table_cells_denominator": expected_cell_count,
        "table_header_exact_match": float(table_rows[0] == expected_rows[0]),
        "table_header_rows_correct": int(table_rows[0] == expected_rows[0]),
        "table_header_rows_denominator": 1,
        "table_row_structure_exact_match": float(table_rows == expected_rows),
        "table_row_structure_correct": int(table_rows == expected_rows),
        "table_row_structure_denominator": 1,
        "verified_table_status_correct": int(table["status"] == "verified"),
        "incomplete_table_status_correct": int(incomplete_table["status"] == "incomplete"),
        "table_status_cases": 2,
        "chart_category_association_accuracy": category_association_accuracy,
        "chart_category_associations_correct": category_matches,
        "chart_category_associations_denominator": 2,
        "estimated_value_pixel_mae": mae if mae is not None else pixel_value_mae,
        "estimated_value_examples": len(errors),
        "estimated_value_expected_examples": len(expected),
        "estimated_value_missing_examples": max(0, len(expected) - len(errors)),
        "estimated_value_mae_tolerance": tolerance,
        "estimated_value_within_tolerance": bool(
            mae is not None and mae <= tolerance and len(errors) == len(expected)
        ),
        "direct_label_accuracy": None,
        "direct_label_examples": 0,
        "provenance_correct": provenance_correct,
        "provenance_denominator": len(provenance_records),
        "incomplete_records": sum(record["status"] == "incomplete" for record in provenance_records),
        "unsupported_records": sum(record["status"] == "unsupported" for record in provenance_records),
        "estimated_records": sum(record["status"] == "estimated" for record in provenance_records),
        "status_record_denominator": len(provenance_records),
        "status_correct": status_correct,
        "status_accuracy": status_correct / len(provenance_records),
        "sample_count": len(provenance_records),
    }


def main() -> int:
    argparse.ArgumentParser(
        description="Run a small synthetic table/chart extraction benchmark."
    ).parse_args()
    result = run_benchmark()
    for metric, value in result.items():
        print(f"{metric}: {value}")
    print("Synthetic fixtures only; these scores do not establish real-document accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
