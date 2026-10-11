from retrieval.hybrid_retriever import (
    reciprocal_rank_fusion,
    select_hybrid_candidates,
)


def _candidate(identifier, source, *, parent=None):
    return {
        "uuid": identifier,
        "type": "Image",
        "content": "",
        "metadata": {
            "document_id": "report",
            "element_id": identifier,
            "parent_id": parent or identifier,
            "root_source_id": parent or identifier,
            "rank_source": source,
        },
    }


def test_rrf_combines_ranks_and_deduplicates_by_stable_element_id():
    text = [_candidate("shared", "text"), _candidate("text-only", "text")]
    visual = [_candidate("visual-only", "visual"), _candidate("shared", "visual")]

    fused = reciprocal_rank_fusion({"text": text, "visual": visual})

    assert [item["uuid"] for item in fused] == ["shared", "visual-only", "text-only"]
    shared = fused[0]
    assert shared["retrieval_sources"] == {"text": 1, "visual": 2}
    assert shared["rrf_score"] == 1 / 61 + 1 / 62


def test_selection_limits_visual_results_and_redundant_tiles_per_parent():
    text = [_candidate("parent-1", "text"), _candidate("text-2", "text")]
    visual = [
        _candidate("tile-1a", "visual", parent="parent-1"),
        _candidate("tile-1b", "visual", parent="parent-1"),
        _candidate("tile-2", "visual", parent="parent-2"),
        _candidate("tile-3", "visual", parent="parent-3"),
    ]

    selected = select_hybrid_candidates(
        text,
        visual,
        text_limit=2,
        visual_limit=2,
    )

    ids = [item["uuid"] for item in selected]
    assert "tile-1a" in ids
    assert "tile-1b" not in ids
    assert len([item for item in selected if "visual" in item["retrieval_sources"]]) == 2
    assert len([item for item in selected if "text" in item["retrieval_sources"]]) == 2


def test_text_results_survive_when_visual_candidate_limit_is_exhausted():
    shared_text = _candidate("parent", "text")
    text = [shared_text]
    visual = [_candidate("parent", "visual")]
    selected = select_hybrid_candidates(
        text,
        visual,
        text_limit=1,
        visual_limit=0,
    )

    assert any(item["uuid"] == "parent" for item in selected)
    assert selected[0]["retrieval_sources"] == {"text": 1}
