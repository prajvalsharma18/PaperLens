"""PaperLens Streamlit workspace. All indexing and generation use the existing CLI."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui import services

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="PaperLens · Document intelligence", page_icon="◫", layout="wide")
st.markdown(
    """<style>
    :root { --pl-blue:#2864dc; --pl-ink:#17243a; --pl-muted:#68758a; --pl-line:#e4e9f1; }
    .pl-brand { color:var(--pl-ink); font-weight:750; letter-spacing:-.04em; font-size:1.28rem; }
    .pl-kicker { color:var(--pl-blue); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .pl-muted { color:var(--pl-muted); }
    .pl-panel { border:1px solid var(--pl-line); border-radius:12px; padding:1rem 1.1rem; background:#fff; }
    </style>""",
    unsafe_allow_html=True,
)

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "last_operation" not in st.session_state:
    st.session_state.last_operation = None


def cli_feedback(result: services.CLIResult) -> None:
    if result.timed_out:
        st.error("The operation timed out. Increase PAPERLENS_CLI_TIMEOUT or retry when dependencies are available.")
    elif result.returncode == 0:
        st.success("Operation completed successfully.")
    else:
        st.error(f"Operation failed with exit code {result.returncode}.")
    if result.stdout:
        st.code(result.stdout, language="text")
    if result.stderr:
        with st.expander("Details and diagnostics"):
            st.code(result.stderr, language="text")


def documents() -> list[dict]:
    try:
        return services.list_documents()
    except OSError as exc:
        st.warning(f"Document store is unavailable: {exc}")
        return []


def choose_document(items: list[dict], label: str = "Document") -> dict | None:
    if not items:
        st.info("No documents are available yet. Upload a PDF from the workspace.")
        return None
    ids = [item["document_id"] for item in items]
    current = st.session_state.get("selected_document")
    if current not in ids:
        current = ids[0]
        st.session_state.selected_document = current
    selected = st.selectbox(
        label,
        ids,
        index=ids.index(current),
        format_func=lambda doc_id: next(
            f'{item["filename"]} · {doc_id}' for item in items if item["document_id"] == doc_id
        ),
        key="selected_document",
    )
    return next(item for item in items if item["document_id"] == selected)


def manifest_records(document: dict) -> list[dict]:
    manifest = document.get("manifest") or {}
    records = manifest.get("structured_evidence", [])
    return records if isinstance(records, list) else []


def render_upload() -> None:
    st.subheader("Add a document")
    st.write("Keep the original PDF in the PaperLens document store, then start ingestion when you are ready.")
    limit = max(1, int(os.getenv("PAPERLENS_MAX_UPLOAD_MB", "100")))
    upload = st.file_uploader("Choose a PDF", type=["pdf"], help=f"Maximum size: {limit} MB")
    if upload:
        content = upload.getvalue()
        st.caption(f"{upload.name} · {len(content) / (1024 * 1024):.2f} MB")
        validation = services.validate_pdf(upload.name, content, limit)
        if validation:
            st.error(validation)
            return
        st.warning("Ingestion creates embeddings and may incur OpenAI API charges. Qdrant and a valid OPENAI_API_KEY are required.")
        with st.form("ingest_pdf_form"):
            index_pages = st.checkbox("Index page images when no extracted figure exists")
            use_ocr = st.checkbox("Run optional Tesseract OCR")
            chunk_visuals = st.checkbox("Create deterministic visual crops for large images")
            confirm = st.checkbox("I’m ready to save this PDF and start ingestion")
            submitted = st.form_submit_button("Save PDF and ingest", type="primary")
        if submitted and not confirm:
            st.warning("Check the confirmation box before starting ingestion.")
        if submitted and confirm:
            try:
                document_id, path = services.save_upload(upload.name, content, limit)
            except (OSError, ValueError) as exc:
                st.error(f"Could not save the PDF: {exc}")
                return
            st.session_state.last_operation = {"document_id": document_id, "state": "ingesting"}
            with st.spinner("Ingesting PDF and indexing text…"):
                result = services.run_cli([
                    "ingest", str(path), "--document-id", document_id,
                    *( ["--index-page-images"] if index_pages else [] ),
                    *( ["--ocr"] if use_ocr else [] ),
                    *( ["--chunk-visuals"] if chunk_visuals else [] ),
                ])
            st.session_state.last_operation = {"document_id": document_id, "state": "complete" if result.returncode == 0 else "failed"}
            cli_feedback(result)
            if result.returncode == 0:
                st.session_state.selected_document = document_id
                st.rerun()
            st.caption(f"Saved upload ID: {document_id}. The original file remains available to retry ingestion.")


def home_page(items: list[dict], qdrant: dict) -> None:
    st.markdown('<p class="pl-kicker">Document intelligence workspace</p>', unsafe_allow_html=True)
    st.title("PaperLens")
    st.write("Explore technical PDFs through grounded answers, page comparisons, and visual evidence.")
    render_upload()
    st.divider()
    st.subheader("Your documents")
    if not items:
        st.info("Your workspace is empty. Add a PDF above to get started.")
    else:
        for item in items:
            with st.container(border=True):
                left, right = st.columns([3, 2])
                left.markdown(f"**{item['filename']}**  \n`{item['document_id']}`")
                page_count = item.get("page_count")
                left.caption(f"{page_count if page_count is not None else 'Unknown'} pages · {item['state']}")
                if item.get("manifest"):
                    if not qdrant["reachable"]:
                        right.warning("Qdrant unavailable; index state unknown")
                    else:
                        for collection in (os.getenv("COLLECTION_NAME", "rag_documents"), os.getenv("VISUAL_COLLECTION_NAME", "rag_visuals")):
                            if collection not in qdrant["collections"]:
                                right.caption(f"{collection}: not initialized")
                            else:
                                count = services.qdrant_document_count(collection, item["document_id"])
                                state = "Unknown" if count is None else (f"{count} points" if count else "No points")
                                right.caption(f"{collection}: {state}")
                else:
                    right.info("Saved upload · not ingested")
    with st.sidebar:
        st.markdown('<p class="pl-brand">◫ PaperLens</p>', unsafe_allow_html=True)
        st.caption("Multimodal document intelligence")
        st.markdown("**Services**")
        st.write("Qdrant reachable" if qdrant["reachable"] else "Qdrant unavailable")
        if qdrant["reachable"]:
            st.caption(f"Collections: {', '.join(qdrant['collections']) or 'none'}")


def chat_page(items: list[dict]) -> None:
    st.title("Ask your documents")
    chosen = choose_document(items)
    if not chosen or not chosen.get("manifest"):
        if chosen:
            st.warning("This PDF has not been ingested yet.")
        return
    doc_id = chosen["document_id"]
    st.caption(f"Active document · {chosen['filename']} · {doc_id}")
    examples = [
        "Summarize the main findings in this document.",
        "Compare the architectural differences between page 1 and page 2.",
        "Explain the chart on page 4.",
        "Find the table containing these values.",
        "What does this diagram show?",
        "What is the value of the tallest bar?",
    ]
    with st.expander("Suggested questions"):
        st.write(" · ".join(examples))
    history = st.session_state.chat_history.setdefault(doc_id, [])
    if st.button("Clear this conversation"):
        st.session_state.chat_history[doc_id] = []
        st.rerun()
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("sources"):
                with st.expander("Sources and evidence references"):
                    for source in turn["sources"]:
                        st.markdown(f"- {source}")
    prompt = st.chat_input("Ask a question about this document")
    if prompt:
        st.warning("A query may call OpenAI and incur API charges.")
        with st.spinner("Retrieving evidence and preparing an answer…"):
            result = services.run_cli(["query", prompt, "--document-id", doc_id])
        if result.returncode == 0:
            answer, sources = services.parse_query_output(result.stdout)
            history.append({"question": prompt, "answer": answer, "sources": sources})
            st.rerun()
        else:
            cli_feedback(result)
            st.caption("The previous conversation is unchanged.")


def compare_page(items: list[dict]) -> None:
    st.title("Compare pages")
    chosen = choose_document(items)
    if not chosen or not chosen.get("manifest"):
        return
    page_count = chosen.get("page_count") or 1
    left, right = st.columns(2)
    first = left.number_input("First page", min_value=1, max_value=max(1, page_count), value=1, step=1)
    second = right.number_input("Second page", min_value=1, max_value=max(1, page_count), value=min(2, page_count), step=1)
    question = st.text_input("Comparison focus (optional)", placeholder="e.g. architecture, results, layout")
    st.caption("Page comparisons use the existing query pipeline and may incur OpenAI API charges.")
    if st.button("Compare selected pages", type="primary"):
        if first == second:
            st.error("Choose two different pages to compare.")
        else:
            focus = f" focusing on {question.strip()}" if question.strip() else ""
            query = f"Compare the content and differences between page {int(first)} and page {int(second)}{focus}."
            with st.spinner("Comparing the requested pages…"):
                result = services.run_cli(["query", query, "--document-id", chosen["document_id"]])
            if result.returncode == 0:
                answer, sources = services.parse_query_output(result.stdout)
                st.session_state["compare_result"] = (chosen["document_id"], int(first), int(second), answer, sources)
            cli_feedback(result)
    comparison = st.session_state.get("compare_result")
    if comparison and comparison[0] == chosen["document_id"]:
        _, page_a, page_b, answer, sources = comparison
        st.subheader("Requested pages")
        cols = st.columns(2)
        page_map = {int(p.get("page_number", 0)): p for p in chosen["manifest"].get("pages", []) if isinstance(p, dict)}
        for col, number in zip(cols, (page_a, page_b)):
            col.markdown(f"**Page {number}**")
            image = services.resolve_asset((page_map.get(number) or {}).get("image_path"))
            if image:
                col.image(str(image), use_container_width=True)
            else:
                col.info(f"The render for page {number} is unavailable; no other page was substituted.")
        st.subheader("Comparison answer")
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                st.write(sources)


def visual_page(items: list[dict]) -> None:
    st.title("Visual evidence")
    chosen = choose_document(items)
    if not chosen or not chosen.get("manifest"):
        return
    manifest = chosen["manifest"]
    elements = services.document_elements(chosen["document_id"])
    parents = {str(element.get("uuid")): element for element in elements}
    pages = manifest.get("pages", [])
    chunks = manifest.get("visual_chunks", [])
    available = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        image = services.resolve_asset(page.get("image_path"))
        if image:
            with st.expander(f"Page {page.get('page_number', 'Unknown')} · full-page render", expanded=False):
                st.image(str(image), caption=f"{chosen['filename']} · page {page.get('page_number', 'Unknown')}", use_container_width=True)
            available += 1
    for element in elements:
        metadata = element.get("metadata", {})
        if element.get("type") != "Image":
            continue
        path = services.resolve_asset(metadata.get("image_path"))
        if path:
            title = f"Figure · page {metadata.get('page', 'Unknown')} · {element.get('uuid', 'unknown element')}"
            with st.expander(title):
                st.image(str(path), caption=f"{chosen['filename']} · page {metadata.get('page', 'Unknown')} · element {element.get('uuid')}", use_container_width=True)
                if metadata.get("bbox") is not None:
                    st.caption(f"Bounding box · {metadata['bbox']} · {metadata.get('bbox_coordinate_system', 'coordinate system unavailable')}")
                st.caption(f"Content hash · {metadata.get('content_hash', element.get('content_hash', 'Unknown'))}")
            available += 1
    for tile in chunks if isinstance(chunks, list) else []:
        metadata = tile.get("metadata", {}) if isinstance(tile, dict) else {}
        child = services.resolve_asset(metadata.get("child_image_path"))
        if not child:
            continue
        parent = parents.get(str(metadata.get("parent_id")), {})
        parent_path = services.resolve_asset(parent.get("metadata", {}).get("image_path") or metadata.get("parent_image_path"))
        with st.expander(f"Crop / region · page {metadata.get('page', 'Unknown')} · {tile.get('uuid', 'unknown tile')}"):
            columns = st.columns(2)
            if parent_path:
                columns[0].image(str(parent_path), caption="Original parent image", use_container_width=True)
            else:
                columns[0].info("Parent image unavailable.")
            columns[1].image(str(child), caption="Child crop / region", use_container_width=True)
            columns[1].caption(f"Parent {metadata.get('parent_id', 'Unknown')} · bounds {metadata.get('bbox', 'Unknown')} px")
        available += 1
    if not available:
        st.info("No readable page renders, extracted figures, or crop assets are available in this document manifest.")


def index_page(items: list[dict], qdrant: dict) -> None:
    st.title("Visual indexing")
    st.write("Build the separate visual Qdrant index from already saved figures, opted-in page images, and child crops.")
    chosen = choose_document(items)
    if not chosen or not chosen.get("manifest"):
        return
    if not qdrant["reachable"]:
        st.warning("Qdrant is unavailable. Start the configured Qdrant service before indexing.")
    st.info("This operation generates local CLIP embeddings. The first run may download model weights; it does not call OpenAI.")
    if st.button("Index this document's visual assets", type="primary", disabled=not qdrant["reachable"]):
        with st.spinner("Loading CLIP and indexing saved visual assets…"):
            result = services.run_cli(["index-visuals", "--document-id", chosen["document_id"]])
        cli_feedback(result)
        if result.returncode == 0:
            st.rerun()
    if qdrant["reachable"]:
        visual_name = os.getenv("VISUAL_COLLECTION_NAME", "rag_visuals")
        if visual_name not in qdrant["collections"]:
            st.caption("Visual index not initialized.")
        else:
            count = services.qdrant_document_count(visual_name, chosen["document_id"])
            st.caption(f"Visual points for this document: {count if count is not None else 'Unknown'}")


def structured_page(items: list[dict]) -> None:
    st.title("Tables and chart evidence")
    st.caption("Chart extraction is restricted to explicitly tagged, supported vertical bars and histograms with supplied geometry and axis metadata; digitized values remain estimates.")
    chosen = choose_document(items)
    if not chosen or not chosen.get("manifest"):
        return
    if st.button("Extract structured evidence", type="primary"):
        with st.spinner("Extracting supported table and chart evidence locally…"):
            result = services.run_cli(["extract-data", "--document-id", chosen["document_id"]])
        cli_feedback(result)
        if result.returncode == 0:
            st.session_state["structured_refresh"] = True
            st.rerun()
    records = manifest_records(chosen)
    if not records:
        st.info("No structured evidence is recorded yet. Extraction is local and runs only when requested.")
    for index, record in enumerate(records):
        data = record.get("data", {})
        status = record.get("status", "Unknown")
        with st.container(border=True):
            st.markdown(f"**{record.get('kind', 'Evidence').title()} · {record.get('source_file') or chosen['filename']} · page {record.get('page') or 'Unknown'}**")
            st.caption(f"{status.title()} · {record.get('extraction_method', 'method unknown')} · source {record.get('source_element_id', 'unknown')}")
            if record.get("kind") == "table" and data.get("rows"):
                rows, headers = data["rows"], data.get("headers", [])
                if headers and all(isinstance(row, (list, tuple)) and len(row) == len(headers) for row in rows):
                    st.dataframe([dict(zip(headers, row)) for row in rows], use_container_width=True, hide_index=True)
                else:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
            elif record.get("kind") == "chart":
                st.caption(f"Chart type: {data.get('chart_type', 'Unknown')} · units: {data.get('y_axis_unit') or 'not provided'}")
                if data.get("values"):
                    st.dataframe(data["values"], use_container_width=True, hide_index=True)
            else:
                st.json(data)
            if status == "estimated":
                st.warning("Values are estimates, not directly extracted measurements.")
            for limitation in record.get("limitations", []):
                st.caption(f"Limitation: {limitation}")
            for warning in record.get("warnings", []):
                st.warning(warning)


def diagnostics_page(qdrant: dict) -> None:
    st.title("Settings and diagnostics")
    st.write(f"Python · {Path(sys.executable).name} · {sys.version.split()[0]}")
    st.write(f"Qdrant · {'Reachable' if qdrant['reachable'] else 'Unavailable'}")
    if qdrant["reachable"]:
        st.write(f"Collections · {', '.join(qdrant['collections']) or 'None'}")
        for name in (os.getenv("COLLECTION_NAME", "rag_documents"), os.getenv("VISUAL_COLLECTION_NAME", "rag_visuals")):
            count = services.qdrant_collection_count(name) if name in qdrant["collections"] else None
            st.write(f"{name} · {count if count is not None else ('Not initialized' if name not in qdrant['collections'] else 'Unknown')}")
    st.subheader("Configuration")
    for name, configured in services.config_status().items():
        st.write(f"{name} · {'Configured' if configured else 'Missing'}")
    st.caption("Secret values are never displayed. OpenAI is not contacted for diagnostics.")
    st.subheader("Optional tools")
    for tool in ("pdftoppm", "pdfinfo", "tesseract"):
        st.write(f"{tool} · {'Available' if shutil.which(tool) else 'Unavailable'}")
    st.caption("CLIP weights may download on the first visual-index run if they are not already cached.")
    st.caption("Local launch: .venv\\Scripts\\python.exe -m streamlit run app.py")


qdrant_info = services.qdrant_status()
items = documents()
with st.sidebar:
    st.markdown('<p class="pl-brand">◫ PaperLens</p>', unsafe_allow_html=True)
    st.caption("Document intelligence")
    page = st.radio("Workspace", ["Home", "Chat", "Compare pages", "Visual evidence", "Visual indexing", "Tables & charts", "Diagnostics"], label_visibility="collapsed")

if page == "Home":
    home_page(items, qdrant_info)
elif page == "Chat":
    chat_page(items)
elif page == "Compare pages":
    compare_page(items)
elif page == "Visual evidence":
    visual_page(items)
elif page == "Visual indexing":
    index_page(items, qdrant_info)
elif page == "Tables & charts":
    structured_page(items)
else:
    diagnostics_page(qdrant_info)
