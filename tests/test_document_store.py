from retrieval.document_store import (
    list_document_manifests,
    load_document_manifest,
    load_page_elements,
    load_raw_element,
    save_document_manifest,
    save_raw_element,
)


def test_document_assets_and_manifest_use_configured_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "paperlens-store"))
    element = {
        "uuid": "element-001",
        "type": "NarrativeText",
        "content": "A paragraph",
        "metadata": {
            "document_id": "document-001",
            "source_file": "report.pdf",
            "page": 2,
        },
    }
    manifest = {
        "document_id": "document-001",
        "source_file": "report.pdf",
        "pages": [{"page_number": 2, "image_path": "assets/document/page-00002.png"}],
    }

    save_raw_element(element)
    manifest_path = save_document_manifest("document-001", manifest)

    assert manifest_path.is_file()
    assert load_raw_element("element-001") == element
    assert load_document_manifest("document-001") == manifest


def test_manifest_reingestion_replaces_same_document_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "paperlens-store"))

    save_document_manifest("same-document", {"page_count": 1})
    save_document_manifest("same-document", {"page_count": 2})

    assert load_document_manifest("same-document") == {"page_count": 2}


def test_page_element_lookup_and_manifest_listing_are_document_scoped(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DOC_STORE_PATH", str(tmp_path / "paperlens-store"))
    matching = {
        "uuid": "matching",
        "type": "Title",
        "content": "Page 2 title",
        "metadata": {"document_id": "report-a", "page": 2},
    }
    other_document = {
        "uuid": "other-document",
        "type": "Title",
        "content": "Other report",
        "metadata": {"document_id": "report-b", "page": 2},
    }
    other_page = {
        "uuid": "other-page",
        "type": "Title",
        "content": "Page 3 title",
        "metadata": {"document_id": "report-a", "page": 3},
    }
    for element in (matching, other_document, other_page):
        save_raw_element(element)
    save_document_manifest("report-a", {"document_id": "report-a", "page_count": 3})
    save_document_manifest("report-b", {"document_id": "report-b", "page_count": 2})

    assert load_page_elements("report-a", 2) == [matching]
    assert {
        manifest["document_id"] for manifest in list_document_manifests()
    } == {"report-a", "report-b"}
