"""Command-line entry point for PaperLens PDF ingestion and querying."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

from generation.rag_chain import generate_answer
from ingestion.image_summariser import summarise_images
from ingestion.partitioner import partition_pdf
from ingestion.table_converter import convert_tables
from retrieval.document_store import save_raw_element
from retrieval.embedder import get_embedder
from retrieval.multi_vector_retriever import retrieve
from retrieval.vector_store import _get_client, get_vector_store, upsert_documents


class PaperLensError(Exception):
    """An actionable error that can be reported without a traceback."""


def _validate_configuration() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your-openai-api-key-here":
        raise PaperLensError(
            "OPENAI_API_KEY is missing. Set it in the environment or in a local .env file."
        )

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
        }
        element_type = element.get("type")
        content = element.get("content", "")

        if element_type == "Table":
            if not element.get("metadata", {}).get("text_as_html") and not content.strip():
                skipped["empty table"] += 1
                continue
            tables.append(element)
        elif element_type == "Image":
            image_path = element.get("metadata", {}).get("image_path")
            if not image_path or not Path(image_path).is_file():
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


def ingest_pdf(pdf_path: str, document_id: str | None = None) -> int:
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
        elements = partition_pdf(str(path))
    except Exception as exc:
        raise PaperLensError(f"PDF parsing failed for {path.name}: {exc}") from exc

    assigned_document_id = document_id or str(uuid.uuid4())
    prepared, skipped = _prepare_documents(elements, assigned_document_id)
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
    print(f"Skipped elements: {sum(skipped.values())}")
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
    citations = []
    for element in context:
        metadata = element.get("metadata", {})
        source = metadata.get("source")
        page = metadata.get("page")
        document_id = metadata.get("document_id")
        if not source and not document_id:
            continue
        label = source or f"document {document_id}"
        if page is not None:
            label += f", page {page}"
        if document_id:
            label += f" [document ID: {document_id}]"
        if label not in citations:
            citations.append(label)
    return citations


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

    _validate_configuration()
    _check_query_collection()

    try:
        embedder = get_embedder()
        context = retrieve(question, embedder, k=k, document_id=document_id)
    except Exception as exc:
        raise PaperLensError(
            "Retrieval failed. Check the embedding configuration and Qdrant collection."
        ) from exc

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
            if (not image_path or not Path(image_path).is_file()) and not (
                element.get("vision_summary") or element.get("content")
            ):
                raise PaperLensError(
                    f"Retrieved image {element.get('uuid', '')!r} has no available "
                    "image file or text summary."
                )
            if not image_path or not Path(image_path).is_file():
                print(
                    f"Warning: image {element.get('uuid', '')!r} is unavailable; "
                    "generation will use its text summary.",
                    file=sys.stderr,
                )
        elif element.get("type") == "Table" and not element.get("content"):
            raise PaperLensError(
                f"Retrieved table {element.get('uuid', '')!r} has no available content."
            )

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

    query_parser = commands.add_parser("query", help="Ask a question about indexed PDFs.")
    query_parser.add_argument("question", help="Natural-language question.")
    query_parser.add_argument(
        "--document-id",
        help="Limit retrieval to one document ID printed by the ingest command.",
    )
    query_parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Maximum number of retrieved elements (default: 4).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            ingest_pdf(args.pdf, document_id=args.document_id)
        else:
            query_documents(args.question, document_id=args.document_id, k=args.k)
    except PaperLensError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
