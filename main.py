"""Command-line entry point for PaperLens PDF ingestion and querying."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

from generation.rag_chain import generate_answer
from generation.evidence_builder import build_evidence_package, format_source_citations
from ingestion.assets import resolve_asset_path
from ingestion.data_extractor import extract_document_data
from ingestion.image_summariser import summarise_images
from ingestion.ocr import OcrError, ocr_image
from ingestion.partitioner import make_element_uuid, partition_pdf
from ingestion.page_renderer import PageRenderError, render_pdf_pages
from ingestion.table_converter import convert_tables
from ingestion.visual_chunker import (
    VisualChunkError,
    chunk_image,
    validate_chunking_parameters,
)
from retrieval.document_store import (
    list_document_manifests,
    load_document_manifest,
    load_page_elements,
    load_structured_evidence,
    save_document_manifest,
    save_raw_element,
)
from retrieval.evidence_dedup import attach_source_hashes, deduplicate_evidence
from retrieval.embedder import get_embedder
from retrieval.hybrid_retriever import select_hybrid_candidates
from retrieval.multi_vector_retriever import retrieve
from retrieval.query_router import QueryPlan, QueryRoutingError, create_query_plan
from retrieval.vector_store import _get_client, get_vector_store, upsert_documents
from retrieval.visual_embedder import VisualEmbeddingError
from retrieval.visual_indexer import (
    VisualIndexError,
    index_document_visual_assets,
    search_visual_assets,
)
from retrieval.visual_vector_store import VisualVectorStoreError, visual_collection_exists


class PaperLensError(Exception):
    """An actionable error that can be reported without a traceback."""


def _validate_configuration() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your-openai-api-key-here":
        raise PaperLensError(
            "OPENAI_API_KEY is missing. Set it in the environment or in a local .env file."
        )
    _validate_qdrant_configuration()


def _validate_qdrant_configuration() -> None:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    if not qdrant_url:
        raise PaperLensError(
            "QDRANT_URL is empty. Set it to your Qdrant URL, for example "
            "http://localhost:6333."
        )


def _validate_pdf(pdf_path: str) -> Path:
    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise PaperLensError(f"PDF file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise PaperLensError(f"Input must have a .pdf extension: {path}")

    try:
        with path.open("rb") as pdf_file:
            signature = pdf_file.read(1024)
    except OSError as exc:
        raise PaperLensError(f"Could not read PDF file {path}: {exc}") from exc

    if b"%PDF-" not in signature:
        raise PaperLensError(f"Input does not have a valid PDF signature: {path}")
    return path.resolve()


def _default_document_id(path: Path) -> str:
    digest = _file_sha256(path)
    identity = f"paperlens:{str(path).casefold()}:{digest}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_documents(
    elements: list[dict[str, Any]],
    document_id: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    text_elements: list[dict[str, Any]] = []

    for element in elements:
        if not element.get("uuid"):
            raise PaperLensError("Partitioner returned an element without a UUID.")
        element["metadata"] = {
            **element.get("metadata", {}),
            "document_id": document_id,
            "element_id": element["uuid"],
        }
        if not element["metadata"].get("content_hash"):
            raw_content = str(element.get("content", ""))
            html = str(element["metadata"].get("text_as_html", ""))
            asset_hash = ""
            image_path = element["metadata"].get("image_path")
            if image_path:
                asset_path = resolve_asset_path(image_path)
                if asset_path.is_file():
                    asset_hash = _file_sha256(asset_path)
            element["metadata"]["content_hash"] = hashlib.sha256(
                f"{element.get('type', '')}\0{raw_content}\0{html}\0{asset_hash}".encode(
                    "utf-8"
                )
            ).hexdigest()
        element_type = element.get("type")
        content = element.get("content", "")

        if element_type == "Table":
            if not element.get("metadata", {}).get("text_as_html") and not content.strip():
                skipped["empty table"] += 1
                continue
            tables.append(element)
        elif element_type == "Image":
            image_path = element.get("metadata", {}).get("image_path")
            if not image_path or not resolve_asset_path(image_path).is_file():
                skipped["image file missing or unavailable"] += 1
                continue
            images.append(element)
        elif element_type in ("Title", "NarrativeText"):
            if not content.strip():
                skipped["empty text element"] += 1
                continue
            text_elements.append(element)
        else:
            skipped[f"unsupported element type: {element_type or 'unknown'}"] += 1

    prepared = convert_tables(tables)
    if images:
        try:
            prepared.extend(summarise_images(images))
        except Exception as exc:
            raise PaperLensError(f"Image summarisation failed: {exc}") from exc
    prepared.extend(text_elements)

    ids = [element["uuid"] for element in prepared]
    if len(ids) != len(set(ids)):
        raise PaperLensError("Partitioner returned duplicate element UUIDs.")
    return prepared, skipped


def _chunk_visual_parents(
    elements: list[dict[str, Any]],
    document_id: str,
    *,
    tile_width: int,
    tile_height: int,
    overlap: int,
    min_image_size: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    validate_chunking_parameters(tile_width, tile_height, overlap, min_image_size)
    chunks: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    parent_count = 0
    for element in elements:
        if element.get("type") != "Image":
            continue
        metadata = element.get("metadata", {})
        image_path = metadata.get("image_path")
        if not image_path:
            continue
        parent_count += 1
        parent_metadata = {
            **metadata,
            "document_id": document_id,
            "element_id": element["uuid"],
            "uuid": element["uuid"],
            "image_path": image_path,
        }
        try:
            result = chunk_image(
                image_path,
                parent_metadata,
                tile_width=tile_width,
                tile_height=tile_height,
                overlap=overlap,
                min_image_size=min_image_size,
            )
        except VisualChunkError as exc:
            raise PaperLensError(
                f"Could not chunk visual parent {element['uuid']!r}: {exc}"
            ) from exc
        chunks.extend(result.chunks)
        stats["assets reused"] += result.assets_reused
        if result.split:
            stats["parents chunked"] += 1
            if parent_metadata.get("extraction_method") == "pypdfium2_page_render":
                stats["page-image parents chunked"] += 1
            else:
                stats["extracted-figure parents chunked"] += 1
        else:
            stats["parents below chunk threshold"] += 1
    return chunks, stats, parent_count


def ingest_pdf(
    pdf_path: str,
    document_id: str | None = None,
    include_page_images: bool = False,
    ocr: bool = False,
    ocr_language: str = "eng",
    render_scale: float = 1.5,
    chunk_visuals: bool = False,
    tile_width: int = 1024,
    tile_height: int = 1024,
    tile_overlap: int = 128,
    min_image_size: int = 1024,
) -> int:
    """Ingest and index one PDF; return the number of indexed elements."""
    path = _validate_pdf(pdf_path)
    _validate_configuration()

    try:
        embedder = get_embedder()
    except Exception as exc:
        raise PaperLensError(
            "Could not initialize the embedding client. Check OPENAI_API_KEY and "
            "EMBED_MODEL."
        ) from exc

    try:
        get_vector_store(embedder)
    except Exception as exc:
        raise PaperLensError(
            "Could not connect to or prepare the Qdrant collection. Check QDRANT_URL, "
            "QDRANT_API_KEY, and that Qdrant is running."
        ) from exc

    try:
        assigned_document_id = document_id or _default_document_id(path)
    except OSError as exc:
        raise PaperLensError(f"Could not calculate a stable document ID: {exc}") from exc
    try:
        elements = partition_pdf(str(path), document_id=assigned_document_id)
    except Exception as exc:
        raise PaperLensError(f"PDF parsing failed for {path.name}: {exc}") from exc

    try:
        rendered_pages = render_pdf_pages(
            str(path), assigned_document_id, scale=render_scale
        )
    except (PageRenderError, ValueError) as exc:
        raise PaperLensError(f"PDF page rendering failed: {exc}") from exc

    page_assets = {page["page_number"]: page for page in rendered_pages}
    extracted_image_pages: set[int] = set()
    text_pages: set[int] = set()
    for element in elements:
        metadata = element.setdefault("metadata", {})
        metadata.setdefault("source", path.name)
        metadata.setdefault("source_file", path.name)
        metadata.setdefault("source_path", str(path))
        page_number = metadata.get("page")
        if page_number not in page_assets:
            continue
        page_asset = page_assets[page_number]
        metadata.setdefault("page_image_path", page_asset["image_path"])
        metadata.setdefault("page_pixel_width", page_asset["pixel_width"])
        metadata.setdefault("page_pixel_height", page_asset["pixel_height"])
        metadata.setdefault("page_width", page_asset["pixel_width"])
        metadata.setdefault("page_height", page_asset["pixel_height"])
        metadata.setdefault("page_dimension_units", "pixels")
        metadata.setdefault("pdf_page_width", page_asset["page_width"])
        metadata.setdefault("pdf_page_height", page_asset["page_height"])
        if "coordinates" not in metadata:
            metadata.setdefault("coordinate_system", page_asset["coordinate_system"])
        metadata.setdefault("element_id", element.get("uuid"))
        if element.get("type") == "Image":
            image_path = metadata.get("image_path")
            if image_path and resolve_asset_path(image_path).is_file():
                extracted_image_pages.add(page_number)
        if element.get("type") in ("Title", "NarrativeText", "Table") and str(
            element.get("content", "")
        ).strip():
            text_pages.add(page_number)
    parser_text_pages = set(text_pages)

    ocr_pages: set[int] = set()
    if ocr:
        try:
            for page_number, page_asset in page_assets.items():
                if page_number in text_pages:
                    continue
                ocr_result = ocr_image(
                    str(resolve_asset_path(page_asset["image_path"])),
                    page_number,
                    language=ocr_language,
                )
                if not ocr_result:
                    continue
                content = ocr_result["text"]
                elements.append(
                    {
                        "uuid": make_element_uuid(
                            assigned_document_id,
                            page_number,
                            "NarrativeText",
                            content,
                        ),
                        "type": "NarrativeText",
                        "content": content,
                        "metadata": {
                            "source": path.name,
                            "source_file": path.name,
                            "source_path": str(path),
                            "page": page_number,
                            "element_type": "NarrativeText",
                            "bbox": ocr_result["bbox"],
                            "page_width": ocr_result["page_width"],
                            "page_height": ocr_result["page_height"],
                            "coordinate_system": ocr_result["coordinate_system"],
                            "page_image_path": page_asset["image_path"],
                            "page_pixel_width": page_asset["pixel_width"],
                            "page_pixel_height": page_asset["pixel_height"],
                            "extraction_method": ocr_result["extraction_method"],
                        },
                    }
                )
                text_pages.add(page_number)
                ocr_pages.add(page_number)
        except OcrError as exc:
            raise PaperLensError(str(exc)) from exc

    if include_page_images:
        for page_number, page_asset in page_assets.items():
            if page_number in extracted_image_pages:
                continue
            image_uuid = make_element_uuid(
                assigned_document_id,
                page_number,
                "Image",
                page_asset["image_path"],
            )
            elements.append(
                {
                    "uuid": image_uuid,
                    "type": "Image",
                    "content": "",
                    "metadata": {
                        "source": path.name,
                        "source_file": path.name,
                        "source_path": str(path),
                        "page": page_number,
                        "element_type": "Image",
                        "image_path": page_asset["image_path"],
                        "original_asset_path": page_asset["image_path"],
                        "page_image_path": page_asset["image_path"],
                        "page_pixel_width": page_asset["pixel_width"],
                        "page_pixel_height": page_asset["pixel_height"],
                        "page_width": page_asset["pixel_width"],
                        "page_height": page_asset["pixel_height"],
                        "page_dimension_units": "pixels",
                        "pdf_page_width": page_asset["page_width"],
                        "pdf_page_height": page_asset["page_height"],
                        "coordinate_system": page_asset["coordinate_system"],
                        "extraction_method": "pypdfium2_page_render",
                    },
                }
            )

    for element in elements:
        page_number = element.get("metadata", {}).get("page")
        if page_number in page_assets:
            page_asset = page_assets[page_number]
            element["metadata"].setdefault("page_image_path", page_asset["image_path"])
            element["metadata"].setdefault("page_pixel_width", page_asset["pixel_width"])
            element["metadata"].setdefault("page_pixel_height", page_asset["pixel_height"])
            element["metadata"].setdefault("page_width", page_asset["pixel_width"])
            element["metadata"].setdefault("page_height", page_asset["pixel_height"])
            element["metadata"].setdefault("page_dimension_units", "pixels")
            element["metadata"].setdefault("pdf_page_width", page_asset["page_width"])
            element["metadata"].setdefault("pdf_page_height", page_asset["page_height"])

    visual_chunks: list[dict[str, Any]] = []
    chunk_stats: Counter[str] = Counter()
    visual_parent_count = 0
    if chunk_visuals:
        try:
            (
                visual_chunks,
                chunk_stats,
                visual_parent_count,
            ) = _chunk_visual_parents(
                elements,
                assigned_document_id,
                tile_width=tile_width,
                tile_height=tile_height,
                overlap=tile_overlap,
                min_image_size=min_image_size,
            )
        except VisualChunkError as exc:
            raise PaperLensError(f"Invalid visual chunking configuration: {exc}") from exc

    try:
        existing_manifest = load_document_manifest(assigned_document_id)
    except (OSError, ValueError) as exc:
        raise PaperLensError(
            f"Could not load existing document provenance manifest: {exc}"
        ) from exc
    manifest_chunks = (
        visual_chunks
        if chunk_visuals
        else (existing_manifest or {}).get("visual_chunks", [])
    )
    try:
        source_content_hash = _file_sha256(path)
    except OSError as exc:
        raise PaperLensError(f"Could not calculate the PDF content hash: {exc}") from exc

    try:
        save_document_manifest(
            assigned_document_id,
            {
                "document_id": assigned_document_id,
                "source_file": path.name,
                "source_path": str(path),
                "source_content_hash": source_content_hash,
                "page_count": len(rendered_pages),
                "visual_chunks": manifest_chunks,
                "pages": [
                    {
                        **page,
                        "has_extracted_image": page["page_number"]
                        in extracted_image_pages,
                        "has_parser_text": page["page_number"] in parser_text_pages,
                        "has_extracted_text": page["page_number"] in text_pages,
                        "ocr_applied": page["page_number"] in ocr_pages,
                    }
                    for page in rendered_pages
                ],
            },
        )
    except OSError as exc:
        raise PaperLensError(f"Could not save document provenance manifest: {exc}") from exc

    prepared, skipped = _prepare_documents(elements, assigned_document_id)
    for element in prepared:
        element.setdefault("metadata", {})["source_content_hash"] = source_content_hash
    if not prepared:
        reasons = ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items()))
        raise PaperLensError(
            f"No indexable elements were extracted from {path.name}. Skipped: "
            f"{reasons or 'no supported elements found'}."
        )

    try:
        for element in prepared:
            save_raw_element(element)
    except Exception as exc:
        raise PaperLensError(f"Could not save original document elements: {exc}") from exc

    try:
        upsert_documents(prepared, embedder)
    except Exception as exc:
        raise PaperLensError(
            "Embedding or Qdrant indexing failed. Original elements were saved, "
            "but no successful index is guaranteed."
        ) from exc

    print(f"Document ID: {assigned_document_id}")
    print(f"Extracted elements: {len(elements)}")
    print(f"Indexed elements: {len(prepared)}")
    print(f"Rendered pages: {len(rendered_pages)}")
    print(f"Skipped elements: {sum(skipped.values())}")
    if chunk_visuals:
        print(f"Visual parents discovered: {visual_parent_count}")
        print(f"Visual parents chunked: {chunk_stats['parents chunked']}")
        print(f"Visual child chunks generated: {len(visual_chunks)}")
        print(f"Visual chunk assets reused: {chunk_stats['assets reused']}")
        print(
            "Visual parents below chunk threshold: "
            f"{chunk_stats['parents below chunk threshold']}"
        )
        print(
            "Extracted-figure parents chunked: "
            f"{chunk_stats['extracted-figure parents chunked']}"
        )
        print(
            "Page-image parents chunked: "
            f"{chunk_stats['page-image parents chunked']}"
        )
    for reason, count in sorted(skipped.items()):
        print(f"  - {reason}: {count}")
    return len(prepared)


def _check_query_collection() -> None:
    collection_name = os.getenv("COLLECTION_NAME", "rag_documents")
    try:
        collection_names = {
            collection.name for collection in _get_client().get_collections().collections
        }
    except Exception as exc:
        raise PaperLensError(
            "Could not reach Qdrant. Check QDRANT_URL, QDRANT_API_KEY, and that "
            "the service is running."
        ) from exc
    if collection_name not in collection_names:
        raise PaperLensError(
            f"Qdrant collection {collection_name!r} does not exist. Ingest a PDF first "
            "or select the correct COLLECTION_NAME."
        )


def _source_citations(context: list[dict[str, Any]]) -> list[str]:
    return format_source_citations(build_evidence_package(context))


def _resolve_page_document(plan: QueryPlan) -> tuple[str, dict[str, Any]]:
    try:
        if plan.document_id:
            manifest = load_document_manifest(plan.document_id)
            if not isinstance(manifest, dict):
                raise PaperLensError(
                    f"Document ID {plan.document_id!r} has no provenance manifest. "
                    "Confirm the ID or re-ingest the PDF."
                )
            if manifest.get("document_id") != plan.document_id:
                raise PaperLensError(
                    f"The provenance manifest does not match document ID "
                    f"{plan.document_id!r}."
                )
            return plan.document_id, manifest

        manifests = list_document_manifests()
    except (OSError, ValueError) as exc:
        raise PaperLensError(f"Could not read document provenance: {exc}") from exc

    if not manifests:
        raise PaperLensError(
            "A page-specific question requires an ingested document manifest. "
            "Ingest the PDF and specify its document ID."
        )
    if len(manifests) > 1:
        raise PaperLensError(
            "A page-specific question is ambiguous because multiple documents are "
            "available. Specify --document-id."
        )

    manifest = manifests[0]
    return str(manifest["document_id"]), manifest


def _query_guidance(plan: QueryPlan) -> list[dict[str, Any]]:
    guidance: list[dict[str, Any]] = []
    if plan.chart_digitization_desirable:
        guidance.append(
            {
                "type": "NarrativeText",
                "metadata": {
                    "query_intent": plan.intent,
                    "chart_digitization_desirable": True,
                },
                "content": (
                    "Use only structured values explicitly present in the retrieved "
                    "evidence. Distinguish verified table cells, OCR-transcribed "
                    "labels, and estimated chart values. Do not infer missing values "
                    "from visual appearance; state when the requested calculation "
                    "cannot be supported."
                ),
            }
        )
    if plan.visual_question and not plan.chart_digitization_desirable:
        guidance.append(
            {
                "type": "NarrativeText",
                "metadata": {"query_intent": plan.intent},
                "content": (
                    "Visual search is limited to available indexed descriptions and "
                    "whole-image evidence. Child tiles are not searchable and there "
                    "is no region detector; if the requested feature is not clearly "
                    "visible, say it could not be identified."
                ),
            }
        )
    return guidance


def _structured_numeric_answer(
    question: str,
    plan: QueryPlan,
) -> str | None:
    if not plan.chart_digitization_desirable or plan.retrieval_strategy == "exact_page":
        return None
    try:
        if plan.document_id:
            manifest = load_document_manifest(plan.document_id)
            if (
                manifest is not None
                and manifest.get("document_id") != plan.document_id
            ):
                raise PaperLensError(
                    "The provenance manifest does not match the requested document ID."
                )
            manifests = [manifest] if manifest else []
        else:
            manifests = list_document_manifests()
        if len(manifests) > 1:
            raise PaperLensError(
                "A numerical visual question is ambiguous across multiple documents. "
                "Specify --document-id."
            )
    except (OSError, ValueError) as exc:
        raise PaperLensError(f"Could not read structured evidence: {exc}") from exc

    if not manifests:
        return None
    if len(manifests) != 1:
        raise PaperLensError(
            "A numerical visual question is ambiguous across multiple documents. "
            "Specify --document-id."
        )
    document_id = manifests[0].get("document_id")
    if not document_id:
        return None
    try:
        records = load_structured_evidence(str(document_id))
    except (OSError, ValueError) as exc:
        raise PaperLensError(f"Could not load structured evidence: {exc}") from exc
    from retrieval.structured_answers import answer_from_structured_data

    answer = answer_from_structured_data(question, records)
    if answer is not None:
        return answer
    return (
        "I cannot provide a numerical answer because the available structured "
        "evidence does not support an unambiguous calculation. Run "
        f"`PaperLens extract-data --document-id {document_id}` and verify the "
        "table or chart extraction; no values were inferred from appearance."
    )


def _page_evidence(plan: QueryPlan) -> list[dict[str, Any]]:
    document_id, manifest = _resolve_page_document(plan)
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise PaperLensError(
            f"Document {document_id!r} has no page-render records. Re-ingest the PDF."
        )
    page_records: dict[int, dict[str, Any]] = {}
    for record in pages:
        if not isinstance(record, dict):
            continue
        page_number = record.get("page_number")
        if isinstance(page_number, int) and not isinstance(page_number, bool):
            page_records[page_number] = record
    page_count = manifest.get("page_count")
    if not isinstance(page_count, int) or isinstance(page_count, bool):
        page_count = max(page_records, default=0)

    for page_number in plan.page_numbers:
        if page_number < 1:
            raise PaperLensError("Page numbers start at 1.")
        if page_number > page_count:
            raise PaperLensError(
                f"Page {page_number} is unavailable in document {document_id!r}; "
                f"the document has {page_count} page(s)."
            )
        if page_number not in page_records:
            raise PaperLensError(
                f"Page {page_number} has no provenance record in document "
                f"{document_id!r}."
            )
        image_path = page_records[page_number].get("image_path")
        if not image_path:
            raise PaperLensError(
                f"Page {page_number} has no rendered image in document "
                f"{document_id!r}. Re-ingest the PDF to render its pages."
            )
        try:
            if not resolve_asset_path(image_path).is_file():
                raise PaperLensError(
                    f"The rendered image for page {page_number} is unavailable in "
                    f"document {document_id!r}."
                )
        except ValueError as exc:
            raise PaperLensError(
                f"The rendered image reference for page {page_number} is invalid."
            ) from exc

    source_file = manifest.get("source_file")
    context: list[dict[str, Any]] = []
    try:
        for page_number in plan.page_numbers:
            page_record = page_records[page_number]
            page_label = f"Document {document_id}, page {page_number}"
            metadata = {
                "document_id": document_id,
                "source": source_file,
                "source_file": source_file,
                "source_path": manifest.get("source_path"),
                "page": page_number,
                "image_path": page_record["image_path"],
                "evidence_type": "full_page_render",
                "evidence_label": f"{page_label} (full-page render)",
                "require_image_evidence": True,
            }
            context.append(
                {
                    "uuid": f"{document_id}:page:{page_number}",
                    "type": "Image",
                    "content": "",
                    "metadata": metadata,
                }
            )
            for element in load_page_elements(document_id, page_number):
                element_metadata = dict(element.get("metadata", {}))
                element_type = element.get("type", "document element")
                element_metadata.update(
                    {
                        "document_id": document_id,
                        "source": element_metadata.get("source") or source_file,
                        "source_file": element_metadata.get("source_file") or source_file,
                        "page": page_number,
                        "evidence_type": "page_element",
                        "evidence_label": (
                            f"{page_label}, {element_type.lower()} "
                            f"{element.get('uuid', '')}"
                        ),
                    }
                )
                if element_type == "Image":
                    image_path = element_metadata.get("image_path")
                    if image_path:
                        try:
                            element_metadata["require_image_evidence"] = (
                                resolve_asset_path(image_path).is_file()
                            )
                        except ValueError:
                            element_metadata.pop("image_path", None)
                context.append({**element, "metadata": element_metadata})
    except (OSError, ValueError) as exc:
        raise PaperLensError(
            f"Could not load original elements for document {document_id!r}: {exc}"
        ) from exc

    return context + _query_guidance(plan)


def _visual_evidence_items(
    candidates: list[dict[str, Any]],
    existing_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def image_key(path: str) -> str:
        try:
            return str(resolve_asset_path(path))
        except ValueError:
            return path

    context: list[dict[str, Any]] = []
    known_images = {
        image_key(str(item.get("metadata", {}).get("image_path")))
        for item in existing_context
        if item.get("metadata", {}).get("image_path")
    }
    for candidate in candidates:
        metadata = dict(candidate.get("metadata", {}))
        image_path = metadata.get("image_path")
        if not image_path:
            continue
        try:
            crop_path = resolve_asset_path(image_path)
        except ValueError:
            print(
                f"Warning: visual result {candidate.get('uuid')!r} has an invalid "
                "asset reference; skipping it.",
                file=sys.stderr,
            )
            continue
        if not crop_path.is_file():
            print(
                f"Warning: visual result {candidate.get('uuid')!r} has no available "
                "image asset; skipping it.",
                file=sys.stderr,
            )
            continue
        crop_key = str(crop_path)

        document_id = metadata.get("document_id", "unknown document")
        page_number = metadata.get("page_number", metadata.get("page"))
        page_text = f", page {page_number}" if page_number is not None else ""
        is_tile = metadata.get("chunk_type") == "image_tile"
        if is_tile:
            label = (
                f"Visual match: document {document_id}{page_text}, child crop of "
                f"element {metadata.get('parent_id')} "
                f"(crop bounds {metadata.get('bbox')} relative to its parent image)"
            )
        else:
            label = f"Visual match: document {document_id}{page_text}, original image"
        metadata.update(
            {
                "image_path": str(image_path),
                "evidence_type": "visual_child_crop" if is_tile else "visual_parent",
                "evidence_label": label,
                "require_image_evidence": True,
            }
        )
        if crop_key not in known_images:
            context.append(
                {
                    **candidate,
                    "type": "Image",
                    "metadata": metadata,
                }
            )
        known_images.add(crop_key)

        parent_path = metadata.get("parent_asset_path")
        if not is_tile or not parent_path:
            continue
        try:
            resolved_parent = resolve_asset_path(parent_path)
        except ValueError:
            continue
        parent_key = str(resolved_parent)
        if not resolved_parent.is_file() or parent_key in known_images:
            continue
        parent_metadata = {
            **metadata,
            "image_path": str(parent_path),
            "chunk_type": metadata.get("parent_type"),
            "bbox": None,
            "bbox_coordinate_system": None,
            "evidence_type": "visual_parent_context",
            "evidence_label": (
                f"Parent image context: document {document_id}{page_text}, "
                f"element {metadata.get('parent_id')}"
            ),
            "require_image_evidence": True,
        }
        context.append(
            {
                "uuid": f"{metadata.get('parent_id')}:visual-parent-context",
                "type": "Image",
                "content": "",
                "metadata": parent_metadata,
            }
        )
        known_images.add(parent_key)
    return context


def _visual_evidence_limit() -> int:
    raw_limit = os.getenv("VISUAL_EVIDENCE_LIMIT", "4")
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise PaperLensError(
            "VISUAL_EVIDENCE_LIMIT must be a positive integer."
        ) from exc
    if limit < 1:
        raise PaperLensError("VISUAL_EVIDENCE_LIMIT must be a positive integer.")
    return limit


def _source_content_hashes() -> dict[str, str]:
    """Read PDF hashes from manifests for conservative cross-ingest deduping."""
    return {
        str(manifest["document_id"]): str(manifest["source_content_hash"])
        for manifest in list_document_manifests()
        if manifest.get("document_id") and manifest.get("source_content_hash")
    }


def _retrieval_status_evidence(status: str, candidate_count: int = 0) -> dict[str, Any]:
    messages = {
        "visual_candidates": (
            f"Visual retrieval returned {candidate_count} ranked candidate(s). "
            "Similarity rank is not a calibrated confidence score. A missing "
            "object in these candidates does not prove it is absent from the full index."
        ),
        "no_visual_candidates": (
            "Visual retrieval completed but returned no usable candidates. This "
            "does not establish that the requested subject is absent from the full index."
        ),
        "visual_index_missing": (
            "No visual collection is available, so CLIP visual retrieval was not run. "
            "Do not describe the images as visually searched."
        ),
        "visual_failed": (
            "CLIP visual retrieval failed. Only semantic text results are available; "
            "do not describe this as a successful visual search."
        ),
        "visual_assets_unavailable": (
            "Visual candidates were returned, but their image assets could not be "
            "loaded. No image inspection was performed for those candidates."
        ),
    }
    return {
        "uuid": f"retrieval-status:{status}",
        "type": "NarrativeText",
        "content": messages[status],
        "metadata": {"retrieval_status": status, "evidence_type": "retrieval_diagnostic"},
    }


def _log_retrieval_diagnostics(
    plan: QueryPlan,
    context: list[dict[str, Any]],
    *,
    visual_status: str,
    fallback: str | None,
    candidate_counts: dict[str, int],
    modalities_run: list[str],
    modalities_attempted: list[str],
) -> None:
    evidence = []
    for rank, item in enumerate(context, start=1):
        metadata = item.get("metadata", {})
        if metadata.get("evidence_type") == "retrieval_diagnostic":
            continue
        sources = metadata.get("retrieval_sources", item.get("retrieval_sources", {}))
        modalities = sorted(sources) if isinstance(sources, dict) else []
        if not modalities:
            modalities = ["page"] if plan.retrieval_strategy == "exact_page" else ["text"]
        evidence.append({
            "document_id": metadata.get("document_id"),
            "page": metadata.get("page", metadata.get("page_number")),
            "element_id": item.get("uuid") or metadata.get("element_id"),
            "evidence_id": metadata.get("evidence_id"),
            "modalities": modalities,
            "rank": metadata.get("visual_rank", rank),
            "visual_score": item.get("visual_score"),
            "rrf_score": metadata.get("rrf_score", item.get("rrf_score")),
            "duplicate_document_ids": metadata.get("duplicate_document_ids", []),
        })
    diagnostic = {
        "event": "paperlens_retrieval",
        "strategy": plan.retrieval_strategy,
        "visual_status": visual_status,
        "fallback": fallback,
        "candidate_counts": candidate_counts,
        "modalities_attempted": modalities_attempted,
        "modalities_run": modalities_run,
        "selected_evidence": evidence,
    }
    print("Retrieval diagnostics: " + json.dumps(diagnostic, ensure_ascii=False), file=sys.stderr)


def _related_structured_evidence(
    plan: QueryPlan,
    context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach chart/table records only when their source element or page is present."""
    element_ids: set[str] = set()
    pages_by_document: dict[str, set[int]] = {}
    documents: set[str] = set()
    for item in context:
        metadata = item.get("metadata", {})
        document_id = metadata.get("document_id")
        if not document_id:
            continue
        document_id = str(document_id)
        documents.add(document_id)
        if metadata.get("page") is not None:
            pages_by_document.setdefault(document_id, set()).add(metadata["page"])
        for identity in (
            item.get("uuid"), metadata.get("element_id"), metadata.get("parent_id")
        ):
            if identity:
                element_ids.add(str(identity))
    if plan.document_id:
        documents.add(plan.document_id)

    existing = {
        str(item.get("metadata", {}).get("record_id"))
        for item in context
        if item.get("metadata", {}).get("record_id")
    }
    related: list[dict[str, Any]] = []
    for document_id in sorted(documents):
        try:
            records = load_structured_evidence(document_id)
        except (OSError, ValueError) as exc:
            raise PaperLensError(
                f"Could not load structured evidence for document {document_id!r}: {exc}"
            ) from exc
        for record in records:
            record_id = str(record.get("record_id", ""))
            if record_id in existing:
                continue
            if (
                str(record.get("source_element_id", "")) not in element_ids
                and record.get("page") not in pages_by_document.get(document_id, set())
            ):
                continue
            metadata = {
                "document_id": document_id,
                "source": record.get("source_file"),
                "source_file": record.get("source_file"),
                "page": record.get("page"),
                "element_id": record.get("source_element_id"),
                "record_id": record_id,
                "extraction_method": record.get("extraction_method"),
                "quality_status": record.get("status"),
                "warnings": record.get("warnings", []),
                "limitations": record.get("limitations", []),
                "evidence_type": f"structured_{record.get('kind')}",
            }
            related.append(
                {
                    "uuid": record_id,
                    "type": "StructuredEvidence",
                    "content": json.dumps(record, ensure_ascii=False, allow_nan=False),
                    "metadata": metadata,
                }
            )
    return related


def query_documents(
    question: str,
    document_id: str | None = None,
    k: int = 4,
) -> str | None:
    """Retrieve context and generate an answer; return None when no context matches."""
    if not question.strip():
        raise PaperLensError("Question cannot be empty.")
    if k < 1:
        raise PaperLensError("The retrieval count must be at least 1.")

    try:
        plan = create_query_plan(question, document_id=document_id)
    except QueryRoutingError as exc:
        raise PaperLensError(str(exc)) from exc
    answer = _structured_numeric_answer(question, plan)
    if answer is not None:
        _log_retrieval_diagnostics(
            plan,
            [],
            visual_status="not_requested",
            fallback="structured_numeric_evidence_short_circuit",
            candidate_counts={"text": 0, "visual": 0},
            modalities_run=[],
            modalities_attempted=[],
        )
        print("Answer:")
        print(answer)
        return answer
    _validate_configuration()

    visual_status = "not_requested"
    visual_fallback: str | None = None
    candidate_counts = {"text": 0, "visual": 0}
    modalities_run: list[str] = []
    modalities_attempted: list[str] = []
    if plan.retrieval_strategy == "exact_page":
        context = _page_evidence(plan)
        modalities_run = ["page"]
        modalities_attempted = ["page"]
        candidate_counts["text"] = len(context)
    else:
        _check_query_collection()
        try:
            embedder = get_embedder()
            retrieval_k = k if plan.chart_digitization_desirable else min(100, k * 4)
            context = retrieve(question, embedder, k=retrieval_k, document_id=document_id)
        except Exception as exc:
            raise PaperLensError(
                "Retrieval failed. Check the embedding configuration and Qdrant collection."
            ) from exc
        modalities_run = ["text"]
        modalities_attempted = ["text"]
        raw_text_candidate_count = len(context)
        source_hashes = _source_content_hashes()
        context = deduplicate_evidence(
            attach_source_hashes(context, source_hashes)
        )[:k]
        candidate_counts["text"] = raw_text_candidate_count
        if plan.visual_question and not plan.chart_digitization_desirable:
            modalities_attempted.append("visual")
            visual_limit = _visual_evidence_limit()
            try:
                if not visual_collection_exists():
                    visual_status = "visual_index_missing"
                    visual_candidates = []
                else:
                    modalities_run.append("visual")
                    visual_candidates = search_visual_assets(
                        question,
                        limit=max(k, visual_limit * 3),
                        document_id=document_id,
                    )
            except (
                VisualEmbeddingError,
                VisualVectorStoreError,
                VisualIndexError,
            ) as exc:
                visual_status = "visual_failed"
                visual_fallback = "text_only_visual_retrieval_failure"
                print(
                    f"Visual retrieval unavailable; using semantic results only: {exc}",
                    file=sys.stderr,
                )
            else:
                visual_candidates = deduplicate_evidence(
                    attach_source_hashes(visual_candidates, source_hashes)
                )
                candidate_counts["visual"] = len(visual_candidates)
                if not visual_candidates:
                    if visual_status != "visual_index_missing":
                        visual_status = "no_visual_candidates"
                    visual_fallback = (
                        "text_only_visual_collection_missing"
                        if visual_status == "visual_index_missing"
                        else "text_only_no_visual_candidates"
                    )
                    if visual_status == "visual_index_missing":
                        print(
                            "Visual collection is missing; using semantic results only.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "Visual retrieval returned no candidates; using semantic results only.",
                            file=sys.stderr,
                        )
                else:
                    candidates = select_hybrid_candidates(
                        context,
                        visual_candidates,
                        text_limit=k,
                        visual_limit=visual_limit,
                    )
                    semantic_context = context
                    selected_text = [
                        candidate
                        for candidate in candidates
                        if "text" in candidate.get("retrieval_sources", {})
                        and "visual" not in candidate.get("retrieval_sources", {})
                    ]
                    visual_ranked = [
                        candidate
                        for candidate in candidates
                        if "visual" in candidate.get("retrieval_sources", {})
                    ]
                    visual_context = _visual_evidence_items(
                        visual_ranked,
                        selected_text,
                    )
                    if visual_context:
                        context = selected_text + visual_context
                        visual_status = "visual_candidates"
                    else:
                        visual_status = "visual_assets_unavailable"
                        visual_fallback = "text_only_visual_assets_unavailable"
                        print(
                            "Visual evidence assets are unavailable; using semantic "
                            "results only.",
                            file=sys.stderr,
                        )
                        context = semantic_context
            context.append(_retrieval_status_evidence(
                visual_status, candidate_counts["visual"]
            ))
        context.extend(_query_guidance(plan))

    if plan.visual_question:
        context.extend(_related_structured_evidence(plan, context))

    _log_retrieval_diagnostics(
        plan,
        context,
        visual_status=visual_status,
        fallback=visual_fallback,
        candidate_counts=candidate_counts,
        modalities_run=modalities_run,
        modalities_attempted=modalities_attempted,
    )

    if (
        plan.chart_digitization_desirable
        and plan.retrieval_strategy != "exact_page"
    ):
        answer = (
            "I cannot provide a numerical answer because the available structured "
            "evidence does not support an unambiguous calculation. No values were "
            "inferred from appearance."
        )
        print("Answer:")
        print(answer)
        return answer

    if not context:
        print("No matching indexed elements were found.")
        return None

    for element in context:
        if element.get("original_element_missing"):
            print(
                f"Warning: original {element.get('type', 'document element')} "
                f"{element.get('uuid', '')!r} is unavailable; using indexed text.",
                file=sys.stderr,
            )
        if element.get("type") == "Image":
            image_path = element.get("metadata", {}).get("image_path")
            if (not image_path or not resolve_asset_path(image_path).is_file()) and not (
                element.get("vision_summary") or element.get("content")
            ):
                raise PaperLensError(
                    f"Retrieved image {element.get('uuid', '')!r} has no available "
                    "image file or text summary."
                )
            if not image_path or not resolve_asset_path(image_path).is_file():
                print(
                    f"Warning: image {element.get('uuid', '')!r} is unavailable; "
                    "generation will use its text summary.",
                    file=sys.stderr,
                )
        elif element.get("type") == "Table" and not element.get("content"):
            raise PaperLensError(
                f"Retrieved table {element.get('uuid', '')!r} has no available content."
            )

    context = build_evidence_package(context)
    try:
        answer = generate_answer(question, context)
    except Exception as exc:
        raise PaperLensError(f"Answer generation failed: {exc}") from exc

    print("Answer:")
    print(answer)
    citations = _source_citations(context)
    if citations:
        print("\nSources:")
        for citation in citations:
            print(f"- {citation}")
    return answer


def index_visuals(
    document_id: str,
    batch_size: int = 16,
) -> int:
    """Embed and upsert existing document images and manifest tiles."""
    if not document_id.strip():
        raise PaperLensError("Document ID cannot be empty.")
    if batch_size < 1:
        raise PaperLensError("Visual indexing batch size must be positive.")
    _validate_qdrant_configuration()
    try:
        result = index_document_visual_assets(
            document_id,
            batch_size=batch_size,
        )
    except (
        OSError,
        ValueError,
        VisualEmbeddingError,
        VisualIndexError,
        VisualVectorStoreError,
    ) as exc:
        raise PaperLensError(f"Visual indexing failed: {exc}") from exc

    print(f"Document ID: {document_id}")
    if result.model_name:
        print(f"Visual model: {result.model_name}")
    if result.vector_dimension:
        print(f"Visual vector dimension: {result.vector_dimension}")
    print(f"Visual parent images upserted: {result.parents_indexed}")
    print(f"Opted-in page images upserted: {result.page_images_indexed}")
    print(f"Visual child tiles upserted: {result.tiles_indexed}")
    print(f"Visual assets skipped: {result.skipped_assets}")
    for reason, count in sorted(result.skip_reasons.items()):
        print(f"  - {count}: {reason}")
    return result.parents_indexed + result.tiles_indexed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PaperLens",
        description="Ingest PDF documents and query their indexed content.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Extract and index a PDF document.")
    ingest_parser.add_argument("pdf", help="Path to a PDF file.")
    ingest_parser.add_argument(
        "--document-id",
        help="Optional ID used to select this document when querying.",
    )
    ingest_parser.add_argument(
        "--index-page-images",
        action="store_true",
        help="Index full-page images only for pages without an extracted figure.",
    )
    ingest_parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run Tesseract OCR on pages without extracted text.",
    )
    ingest_parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Tesseract language code for --ocr (default: eng).",
    )
    ingest_parser.add_argument(
        "--render-scale",
        type=float,
        default=1.5,
        help="Page image scale for PDFium rendering (default: 1.5).",
    )
    ingest_parser.add_argument(
        "--chunk-visuals",
        action="store_true",
        help="Generate spatial image tiles for large extracted figures and opted-in page images.",
    )
    ingest_parser.add_argument(
        "--tile-width",
        type=int,
        default=1024,
        help="Visual tile width in pixels (default: 1024).",
    )
    ingest_parser.add_argument(
        "--tile-height",
        type=int,
        default=1024,
        help="Visual tile height in pixels (default: 1024).",
    )
    ingest_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="Overlap in pixels between adjacent visual tiles (default: 128).",
    )
    ingest_parser.add_argument(
        "--min-image-size",
        type=int,
        default=1024,
        help="Split images when either dimension exceeds this size (default: 1024).",
    )

    query_parser = commands.add_parser("query", help="Ask a question about indexed PDFs.")
    query_parser.add_argument("question", help="Natural-language question.")
    query_parser.add_argument(
        "--document-id",
        help="Select the document for retrieval or explicit page questions.",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Maximum number of retrieved elements (default: 4).",
    )

    visual_parser = commands.add_parser(
        "index-visuals",
        help="Index existing figure and tile assets in the visual collection.",
    )
    visual_parser.add_argument(
        "--document-id",
        required=True,
        help="Document ID printed by the ingest command.",
    )
    visual_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of visual assets per model inference batch (default: 16).",
    )

    extract_parser = commands.add_parser(
        "extract-data",
        help="Extract local table and explicitly tagged bar-chart evidence.",
    )
    extract_parser.add_argument(
        "--document-id",
        required=True,
        help="Document ID printed by the ingest command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            ingest_pdf(
                args.pdf,
                document_id=args.document_id,
                include_page_images=args.index_page_images,
                ocr=args.ocr,
                ocr_language=args.ocr_language,
                render_scale=args.render_scale,
                chunk_visuals=args.chunk_visuals,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                tile_overlap=args.tile_overlap,
                min_image_size=args.min_image_size,
            )
        elif args.command == "query":
            query_documents(args.question, document_id=args.document_id, k=args.k)
        elif args.command == "index-visuals":
            index_visuals(args.document_id, batch_size=args.batch_size)
        else:
            try:
                report = extract_document_data(args.document_id)
            except (OSError, ValueError) as exc:
                raise PaperLensError(f"Structured extraction failed: {exc}") from exc
            print(f"Document ID: {args.document_id}")
            print(f"Structured records: {len(report.records)}")
            for status, count in sorted(report.status_counts.items()):
                print(f"  {status}: {count}")
            for record in report.records:
                if record.get("status") in {"incomplete", "unreadable", "unsupported"}:
                    details = "; ".join(record.get("warnings", []))
                    print(
                        f"{record['status']}: {record['source_element_id']}: {details}",
                        file=sys.stderr,
                    )
            for failure in report.failures:
                print(f"Error: {failure}", file=sys.stderr)
            if report.failures:
                return 1
    except PaperLensError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
