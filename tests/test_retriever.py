"""
Unit tests for the multi-vector retriever.
"""

from unittest.mock import MagicMock, patch


def test_retriever_fetches_raw_for_table():
    """
    When a retrieved document is of type Table, the retriever
    should fetch the raw element from the document store.
    """
    from retrieval.multi_vector_retriever import retrieve

    mock_doc = MagicMock()
    mock_doc.page_content = "Revenue table"
    mock_doc.metadata = {"uuid": "test-uuid-123", "type": "Table"}

    raw_element = {
        "uuid": "test-uuid-123",
        "type": "Table",
        "content": "<table><tr><td>Q3</td><td>$1M</td></tr></table>",
        "metadata": {},
    }

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    with patch("retrieval.vector_store.get_vector_store", return_value=mock_store) as get_store, \
         patch("retrieval.multi_vector_retriever.load_raw_element", return_value=raw_element) as load_raw:
        embedder = MagicMock()
        results = retrieve("What is Q3 revenue?", embedder=embedder)

    get_store.assert_called_once_with(embedder)
    mock_store.similarity_search.assert_called_once_with("What is Q3 revenue?", k=4)
    load_raw.assert_called_once_with("test-uuid-123")
    assert results == [raw_element]


def test_retriever_uses_text_directly_for_narrative():
    """
    NarrativeText documents should be returned directly from the vector store.
    """
    from retrieval.multi_vector_retriever import retrieve

    mock_doc = MagicMock()
    mock_doc.page_content = "The company was founded in 2010."
    mock_doc.metadata = {"uuid": "test-uuid-456", "type": "NarrativeText"}

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [mock_doc]

    with patch("retrieval.vector_store.get_vector_store", return_value=mock_store):
        results = retrieve("When was the company founded?", embedder=MagicMock())

    assert results[0]["content"] == "The company was founded in 2010."


def test_retriever_falls_back_to_vector_document_when_raw_element_is_missing():
    """A missing stored table falls back to its indexed content and metadata."""
    from retrieval.multi_vector_retriever import retrieve

    doc = MagicMock()
    doc.page_content = "Revenue table summary"
    doc.metadata = {"uuid": "missing-uuid", "type": "Table", "page": 3}

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [doc]

    with patch("retrieval.vector_store.get_vector_store", return_value=mock_store), \
         patch("retrieval.multi_vector_retriever.load_raw_element", return_value=None) as load_raw:
        results = retrieve("What is revenue?", embedder=MagicMock())

    load_raw.assert_called_once_with("missing-uuid")
    assert results == [{
        "uuid": "missing-uuid",
        "type": "Table",
        "content": "Revenue table summary",
        "metadata": doc.metadata,
        "original_element_missing": True,
    }]


def test_upsert_preserves_element_uuid_type_and_document_metadata(monkeypatch):
    from retrieval.vector_store import upsert_documents

    store = MagicMock()
    monkeypatch.setattr(
        "retrieval.vector_store.get_vector_store",
        MagicMock(return_value=store),
    )
    element = {
        "uuid": "element-uuid",
        "type": "Table",
        "content": "<table></table>",
        "embed_content": "Searchable table summary",
        "metadata": {"document_id": "report-1", "page": 4},
    }
    embedder = MagicMock()

    upsert_documents([element], embedder)

    store.add_documents.assert_called_once()
    indexed = store.add_documents.call_args.args[0][0]
    assert indexed.page_content == "Searchable table summary"
    assert indexed.metadata == {
        "document_id": "report-1",
        "page": 4,
        "uuid": "element-uuid",
        "type": "Table",
    }
