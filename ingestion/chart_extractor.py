"""Conservative, OCR-assisted extraction for explicitly identified bar charts."""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from ingestion.ocr import OcrError, ocr_image
from ingestion.structured_evidence import make_structured_record


_NUMBER = re.compile(
    r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?)(\s*%)?\s*$"
)
_SUPPORTED_TYPES = {"bar", "histogram"}
_CONFIGURATION_VERSION = "bar-digitizer-v1"


def _box(token: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = token.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    left, top, right, bottom = values
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _parse_number(text: str) -> tuple[float, str | None] | None:
    match = _NUMBER.fullmatch(text.replace("\u2212", "-"))
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return value, "%" if match.group(2) else None


def _incomplete_record(
    element: dict[str, Any],
    document_id: str,
    *,
    chart_type: str | None,
    method: str,
    warning: str,
    data: dict[str, Any] | None = None,
    status: str = "incomplete",
    region: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    metadata = element.get("metadata", {})
    identity = str(element.get("uuid") or metadata.get("element_id") or "")
    return make_structured_record(
        kind="chart",
        document_id=document_id,
        source_element_id=identity,
        source_file=metadata.get("source_file", metadata.get("source")),
        page=metadata.get("page"),
        source_image_path=metadata.get("image_path"),
        source_bbox=metadata.get("bbox"),
        region=region,
        extraction_method=method,
        status=status,
        data=data
        or {
            "chart_type": chart_type,
            "title": metadata.get("chart_title"),
            "x_axis_label": metadata.get("x_axis_label"),
            "y_axis_label": metadata.get("y_axis_label"),
            "x_axis_unit": metadata.get("x_axis_unit"),
            "y_axis_unit": metadata.get("y_axis_unit"),
            "x_scale": metadata.get("x_scale"),
            "y_scale": metadata.get("y_scale"),
            "series_names": metadata.get("series_names", []),
            "category_labels": [],
            "values": [],
        },
        warnings=[warning] if warning else [],
        limitations=limitations or [],
        configuration_version=_CONFIGURATION_VERSION,
    )


def _fit_linear_ticks(
    ticks: list[dict[str, Any]],
) -> tuple[float, float] | None:
    if len(ticks) < 2:
        return None
    ordered = sorted(
        (float(tick["pixel"]), float(tick["value"]))
        for tick in ticks
    )
    positions = [pixel for pixel, _ in ordered]
    values = [value for _, value in ordered]
    if len(set(positions)) < 2 or len(set(values)) < 2:
        return None
    pixel_steps = [
        ordered[i + 1][0] - ordered[i][0]
        for i in range(len(ordered) - 1)
    ]
    value_steps = [
        ordered[i + 1][1] - ordered[i][1]
        for i in range(len(ordered) - 1)
    ]
    if any(step == 0 for step in pixel_steps):
        return None
    directions = {1 if step > 0 else -1 for step in value_steps}
    if len(directions) != 1:
        return None

    mean_pixel = sum(positions) / len(positions)
    mean_value = sum(values) / len(values)
    denominator = sum((pixel - mean_pixel) ** 2 for pixel in positions)
    if denominator == 0:
        return None
    slope = sum(
        (pixel - mean_pixel) * (value - mean_value)
        for pixel, value in zip(positions, values)
    ) / denominator
    intercept = mean_value - slope * mean_pixel
    predicted = [slope * pixel + intercept for pixel in positions]
    residual = max(abs(actual - estimate) for actual, estimate in zip(values, predicted))
    span = max(values) - min(values)
    if residual > max(span * 0.03, 1e-9):
        return None
    return slope, intercept


def _vertical_bar_runs(
    image: Image.Image,
    plot_area: tuple[int, int, int, int],
) -> list[dict[str, int]]:
    left, top, right, bottom = plot_area
    grayscale = image.convert("L")
    pixels = grayscale.load()
    if pixels is None:
        grayscale.close()
        return []
    column_marks: list[tuple[int, int, int] | None] = []
    for x in range(left, right):
        dark_y = []
        for y in range(top, bottom):
            pixel = pixels[x, y]
            if isinstance(pixel, (int, float)) and pixel < 90:
                dark_y.append(y)
        if len(dark_y) >= max(3, math.ceil((bottom - top) * 0.3)):
            column_marks.append((x, min(dark_y), max(dark_y)))
        else:
            column_marks.append(None)
    grayscale.close()

    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    for mark in column_marks:
        if mark is None:
            if current:
                groups.append(current)
                current = []
        else:
            current.append(mark)
    if current:
        groups.append(current)

    bars: list[dict[str, int]] = []
    for group in groups:
        if len(group) < 2:
            continue
        tops = [mark[1] for mark in group]
        bottoms = [mark[2] for mark in group]
        if max(tops) - min(tops) > 3 or max(bottoms) - min(bottoms) > 3:
            continue
        bars.append(
            {
                "x0": group[0][0],
                "x1": group[-1][0] + 1,
                "y0": min(tops),
                "y1": max(bottoms) + 1,
            }
        )
    return bars


def _ocr_tokens(
    element: dict[str, Any],
    *,
    ocr_reader: Callable[[str, int], dict[str, Any] | None] | None,
) -> dict[str, Any] | None:
    metadata = element.get("metadata", {})
    image_path = metadata.get("image_path")
    if not image_path:
        return None
    page = metadata.get("page")
    if not isinstance(page, int):
        page = 0
    if ocr_reader is not None:
        return ocr_reader(str(image_path), page)
    from ingestion.assets import resolve_asset_path

    return ocr_image(str(resolve_asset_path(image_path)), page)


def extract_chart(
    element: dict[str, Any],
    *,
    document_id: str,
    chart_type: str | None = None,
    ocr_reader: Callable[[str, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Extract values only from explicitly typed bar charts with supplied plot bounds.

    Axis digitization is deliberately constrained to linear, non-stacked vertical
    bars and requires OCR tick labels on the left side of `plot_area`.
    """
    metadata = element.get("metadata", {})
    kind = (chart_type or metadata.get("chart_type") or "").strip().lower()
    if kind not in _SUPPORTED_TYPES:
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind or None,
            method="vision_only",
            warning=(
                f"Chart type {kind or 'unspecified'!r} is unsupported; no chart values were inferred."
            ),
            status="unsupported",
            limitations=[
                "Only explicitly identified vertical bar charts and histograms are digitized."
            ],
        )

    scale = str(metadata.get("y_scale", "linear")).lower()
    if scale != "linear":
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning=f"Y-axis scale {scale!r} is not supported for digitization.",
            status="unsupported",
            limitations=["Logarithmic and other nonlinear axes are not digitized."],
        )

    raw_area = metadata.get("plot_area")
    if not isinstance(raw_area, (list, tuple)) or len(raw_area) != 4:
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning="No validated plot_area was supplied; plot boundaries were not guessed.",
        )
    try:
        parsed_area = tuple(int(value) for value in raw_area)
    except (TypeError, ValueError):
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning="The supplied plot_area is invalid.",
        )
    if any(float(raw) != value for raw, value in zip(raw_area, parsed_area)):
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning="The supplied plot_area must use integer pixel coordinates.",
        )
    area_values: tuple[int, int, int, int] = (
        parsed_area[0],
        parsed_area[1],
        parsed_area[2],
        parsed_area[3],
    )
    left, top, right, bottom = area_values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning="The supplied plot_area is outside valid image geometry.",
        )

    from ingestion.assets import resolve_asset_path

    source_image = metadata.get("image_path")
    if not source_image:
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning="The chart has no source image asset.",
        )
    image_path = resolve_asset_path(source_image)
    try:
        with Image.open(image_path) as source:
            source.verify()
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method="vision_only",
            warning=f"The source chart image is unreadable: {exc}",
            status="unreadable",
        )
    try:
        if right > image.width or bottom > image.height:
            return _incomplete_record(
                element,
                document_id,
                chart_type=kind,
                method="vision_only",
                warning="The supplied plot_area extends beyond the source image.",
            )
        try:
            ocr = _ocr_tokens(
                element,
                ocr_reader=ocr_reader,
            )
        except OcrError as exc:
            return _incomplete_record(
                element,
                document_id,
                chart_type=kind,
                method="vision_only",
                warning=str(exc),
                region={"plot_area": list(area_values), "coordinate_system": "image_pixels"},
            )
        except (OSError, ValueError) as exc:
            return _incomplete_record(
                element,
                document_id,
                chart_type=kind,
                method="vision_only",
                warning=f"Chart OCR could not read the source asset: {exc}",
            )
        tokens = ocr.get("tokens", []) if isinstance(ocr, dict) else []
        valid_tokens = [
            {"text": str(token.get("text", "")).strip(), "bbox": _box(token)}
            for token in tokens
            if isinstance(token, dict) and _box(token) is not None
        ]
        tick_records = []
        for token in valid_tokens:
            box = token["bbox"]
            number = _parse_number(token["text"])
            if number is None or box[2] > left or box[3] < top or box[1] > bottom:
                continue
            tick_records.append(
                {
                    "raw_text": token["text"],
                    "value": number[0],
                    "unit": number[1],
                    "pixel": (box[1] + box[3]) / 2.0,
                    "bbox": list(box),
                }
            )
        calibration = _fit_linear_ticks(tick_records)
        bars = _vertical_bar_runs(image, area_values)
        if not bars:
            return _incomplete_record(
                element,
                document_id,
                chart_type=kind,
                method="vision_only",
                warning="No supported vertical bar geometry was detected in the supplied plot_area.",
                data={
                    "chart_type": kind,
                    "title": metadata.get("chart_title"),
                    "x_axis_label": metadata.get("x_axis_label"),
                    "y_axis_label": metadata.get("y_axis_label"),
                    "x_axis_unit": metadata.get("x_axis_unit"),
                    "y_axis_unit": metadata.get("y_axis_unit"),
                    "x_scale": metadata.get("x_scale", "categorical"),
                    "y_scale": scale,
                    "series_names": metadata.get("series_names", []),
                    "category_labels": [],
                    "values": [],
                    "axis_ticks": tick_records,
                },
                region={"plot_area": list(area_values), "coordinate_system": "image_pixels"},
            )

        category_tokens = []
        for token in valid_tokens:
            box = token["bbox"]
            if box[1] < bottom or _parse_number(token["text"]) is not None:
                continue
            category_tokens.append(
                {
                    "text": token["text"],
                    "center_x": (box[0] + box[2]) / 2.0,
                    "bbox": list(box),
                }
            )

        values: list[dict[str, Any]] = []
        direct_any = False
        estimated_any = False
        used_labels: set[int] = set()
        for bar in bars:
            center_x = (bar["x0"] + bar["x1"]) / 2.0
            category_candidates = sorted(
                category_tokens,
                key=lambda token: abs(token["center_x"] - center_x),
            )
            category = (
                category_candidates[0]["text"]
                if category_candidates
                and abs(category_candidates[0]["center_x"] - center_x)
                <= max(24, (bar["x1"] - bar["x0"]) * 2)
                else None
            )
            label_candidates = []
            for token_index, token in enumerate(valid_tokens):
                parsed = _parse_number(token["text"])
                box = token["bbox"]
                token_center_x = (box[0] + box[2]) / 2.0
                token_center_y = (box[1] + box[3]) / 2.0
                if (
                    parsed is not None
                    and token_index not in used_labels
                    and box[0] >= left
                    and box[2] <= right
                    and abs(token_center_x - center_x)
                    <= max(10, (bar["x1"] - bar["x0"]) / 2)
                    and top <= token_center_y <= bar["y0"] + max(30, box[3] - box[1])
                ):
                    label_candidates.append((token_index, parsed, box, token["text"]))
            if len(label_candidates) == 1:
                token_index, parsed, label_box, raw_label = label_candidates[0]
                used_labels.add(token_index)
                direct_any = True
                values.append(
                    {
                        "category": category,
                        "series": metadata.get("series_name"),
                        "value": parsed[0],
                        "unit": parsed[1] or metadata.get("y_axis_unit"),
                        "value_source": "printed_data_label",
                        "extraction_method": "printed_data_label",
                        "directly_transcribed": True,
                        "raw_label": raw_label,
                        "label_bbox": list(label_box),
                        "pixel_geometry": dict(bar),
                        "uncertainty": "OCR label association is not human verified.",
                    }
                )
            elif calibration is not None:
                slope, intercept = calibration
                y_position = float(bar["y0"])
                estimated = slope * y_position + intercept
                estimated_any = True
                values.append(
                    {
                        "category": category,
                        "series": metadata.get("series_name"),
                        "value": estimated,
                        "unit": metadata.get("y_axis_unit")
                        or next((tick["unit"] for tick in tick_records if tick["unit"]), None),
                        "value_source": "axis_digitization",
                        "extraction_method": "axis_digitization",
                        "directly_transcribed": False,
                        "pixel_geometry": dict(bar),
                        "calibration": {
                            "scale": "linear",
                            "slope_value_per_pixel": slope,
                            "intercept": intercept,
                            "reference_ticks": tick_records,
                            "pixel_coordinate": "top-left origin; y increases downward",
                        },
                        "uncertainty": (
                            "Estimated from raster geometry and OCR axis ticks; "
                            "not the exact underlying source value."
                        ),
                    }
                )

        if not values:
            status = "incomplete"
            method = "vision_only"
            warnings = [
                "No directly associated printed value labels or valid linear axis calibration were available."
            ]
        elif estimated_any:
            status = "estimated"
            method = (
                "mixed_printed_and_axis_digitization"
                if direct_any
                else "axis_digitization"
            )
            warnings = []
            if direct_any:
                warnings.append("Printed-label OCR values are not human verified.")
        else:
            status = "incomplete"
            method = "printed_data_label"
            warnings = ["Printed-label OCR values and their bar associations are not human verified."]
        data = {
            "chart_type": kind,
            "title": metadata.get("chart_title"),
            "x_axis_label": metadata.get("x_axis_label"),
            "y_axis_label": metadata.get("y_axis_label"),
            "x_axis_unit": metadata.get("x_axis_unit"),
            "y_axis_unit": metadata.get("y_axis_unit")
            or next((tick["unit"] for tick in tick_records if tick["unit"]), None),
            "x_scale": metadata.get("x_scale", "categorical"),
            "y_scale": scale,
            "series_names": metadata.get("series_names", []),
            "category_labels": [value["category"] for value in values],
            "values": values,
            "axis_ticks": tick_records,
        }
        return _incomplete_record(
            element,
            document_id,
            chart_type=kind,
            method=method,
            warning=warnings[0] if warnings else "",
            data=data,
            status=status,
            region={
                "plot_area": list(area_values),
                "coordinate_system": "image_pixels",
                "crop_bbox": metadata.get("bbox"),
            },
            limitations=[
                "Digitization supports explicitly bounded, non-stacked vertical bars on linear axes only.",
                "OCR labels and bar/category associations require human review before treating them as verified source data.",
            ],
        )
    finally:
        image.close()
