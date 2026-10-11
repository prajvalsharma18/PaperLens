"""Rank fusion helpers for text-description and visual retrieval results."""

from __future__ import annotations

from typing import Any


def _candidate_id(candidate: dict[str, Any]) -> str | None:
    metadata = candidate.get("metadata", {})
    return str(
        candidate.get("uuid")
        or metadata.get("chunk_id")
        or metadata.get("element_id")
        or ""
    ) or None


def reciprocal_rank_fusion(
    rankings: dict[str, list[dict[str, Any]]],
    *,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked lists with RRF while retaining per-list provenance."""
    if rank_constant < 1:
        raise ValueError("RRF rank constant must be positive.")
    fused: dict[str, dict[str, Any]] = {}
    for source, candidates in rankings.items():
        for rank, candidate in enumerate(candidates, start=1):
            identity = _candidate_id(candidate)
            if identity is None:
                continue
            item = fused.get(identity)
            if item is None:
                item = {
                    "candidate": dict(candidate),
                    "rrf_score": 0.0,
                    "retrieval_sources": {},
                }
                fused[identity] = item
            item["rrf_score"] += 1.0 / (rank_constant + rank)
            item["retrieval_sources"][source] = rank
            if source == "visual" and "text" in item["retrieval_sources"]:
                metadata = dict(item["candidate"].get("metadata", {}))
                visual_metadata = candidate.get("metadata", {})
                for key in (
                    "parent_asset_path",
                    "asset_path",
                    "image_path",
                    "bbox",
                    "bbox_coordinate_system",
                    "chunk_type",
                ):
                    if visual_metadata.get(key) is not None:
                        metadata[key] = visual_metadata[key]
                item["candidate"]["metadata"] = metadata

    ordered = sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], _candidate_id(item["candidate"]) or ""),
    )
    return [
        {
            **item["candidate"],
            "rrf_score": item["rrf_score"],
            "retrieval_sources": item["retrieval_sources"],
        }
        for item in ordered
    ]


def select_hybrid_candidates(
    text_candidates: list[dict[str, Any]],
    visual_candidates: list[dict[str, Any]],
    *,
    text_limit: int,
    visual_limit: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse candidate lists with limits on text results and visual parents."""
    fused = reciprocal_rank_fusion(
        {"text": text_candidates, "visual": visual_candidates},
        rank_constant=rank_constant,
    )
    selected: list[dict[str, Any]] = []
    text_count = 0
    visual_count = 0
    selected_visual_parents: set[str] = set()
    for candidate in fused:
        sources = dict(candidate["retrieval_sources"])
        metadata = candidate.get("metadata", {})
        if "visual" in sources:
            parent_id = str(
                metadata.get("root_source_id")
                or metadata.get("parent_id")
                or candidate.get("uuid")
            )
            if (
                visual_count >= visual_limit
                or parent_id in selected_visual_parents
            ):
                sources.pop("visual")
            else:
                visual_count += 1
                selected_visual_parents.add(parent_id)
        if "text" in sources and text_count >= text_limit:
            sources.pop("text")
        if not sources:
            continue
        if "text" in sources:
            text_count += 1
        candidate["retrieval_sources"] = sources
        candidate["metadata"] = {
            **metadata,
            "retrieval_sources": sources,
            "rrf_score": candidate["rrf_score"],
        }
        selected.append(candidate)
    return selected
