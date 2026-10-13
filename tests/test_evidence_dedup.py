from retrieval.evidence_dedup import attach_source_hashes, deduplicate_evidence


def _item(document_id, *, filename="same.pdf", source_hash=None, content_hash="asset-a"):
    metadata = {
        "document_id": document_id,
        "source": filename,
        "page": 4,
        "chunk_type": "image",
        "content_hash": content_hash,
    }
    if source_hash:
        metadata["source_content_hash"] = source_hash
    return {"uuid": document_id + "-element", "type": "Image", "metadata": metadata}


def test_duplicate_evidence_requires_both_source_and_content_hashes():
    same_name = deduplicate_evidence([
        _item("a", filename="same.pdf"),
        _item("b", filename="same.pdf"),
    ])
    different_source = deduplicate_evidence([
        _item("a", source_hash="pdf-a"),
        _item("b", source_hash="pdf-b"),
    ])
    different_content = deduplicate_evidence([
        _item("a", source_hash="pdf-a", content_hash="asset-a"),
        _item("b", source_hash="pdf-a", content_hash="asset-b"),
    ])

    assert len(same_name) == len(different_source) == len(different_content) == 2


def test_byte_identical_document_evidence_collapses_and_keeps_source_ids():
    result = deduplicate_evidence([
        _item("copy-b", source_hash="pdf-hash"),
        _item("copy-a", source_hash="pdf-hash"),
    ])

    assert len(result) == 1
    assert result[0]["metadata"]["document_id"] == "copy-a"
    assert result[0]["metadata"]["duplicate_document_ids"] == ["copy-a", "copy-b"]
    assert result[0]["metadata"]["duplicate_result_count"] == 2


def test_manifest_hashes_are_attached_by_document_identity():
    result = attach_source_hashes([_item("doc")], {"doc": "pdf-hash"})

    assert result[0]["metadata"]["source_content_hash"] == "pdf-hash"
