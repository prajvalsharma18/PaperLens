from evaluation.duplicate_documents import find_duplicate_documents


def test_duplicate_document_report_groups_only_explicit_source_hashes():
    manifests = [
        {"document_id": "copy-b", "source_name": "notes.pdf", "source_content_hash": "same", "page_count": 4},
        {"document_id": "copy-a", "source_name": "renamed.pdf", "source_content_hash": "same", "page_count": 4},
        {"document_id": "other", "source_name": "notes.pdf", "source_content_hash": "different"},
        {"document_id": "legacy", "source_name": "notes.pdf"},
    ]

    report = find_duplicate_documents(manifests)

    assert len(report) == 1
    assert report[0]["source_content_hash"] == "same"
    assert [item["document_id"] for item in report[0]["documents"]] == ["copy-a", "copy-b"]
