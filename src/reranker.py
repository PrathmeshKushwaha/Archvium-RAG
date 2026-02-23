"""
reranker.py - Cross-encoder reranking with FlashRank.

Takes the top-N candidates from hybrid search and returns the best-K
by running a lightweight cross-encoder that scores (query, passage) pairs.
"""

import logging
from dataclasses import dataclass

from flashrank import Ranker, RerankRequest

from src.retrieval import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A SearchResult enriched with a reranker score."""
    search_result: SearchResult
    rerank_score: float


class ChunkReranker:
    """
    Wraps FlashRank to rerank retrieval candidates.

    FlashRank runs fully locally – no network call required.

    Args:
        model_name: FlashRank model identifier.
                    "ms-marco-MiniLM-L-12-v2" is a good balance of speed/quality.
        top_k:      Number of best chunks to keep after reranking.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        top_k: int = 5,
    ) -> None:
        logger.info("Loading FlashRank reranker model '%s' …", model_name)
        self._ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
        self._top_k = top_k
        logger.info("Reranker ready (top_k=%d).", top_k)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[RankedResult]:
        """
        Rerank *candidates* for *query* and return the top-k best matches.

        Args:
            query:      The user's search query.
            candidates: Up to 15 SearchResult objects from hybrid search.

        Returns:
            Up to self.top_k RankedResult objects sorted by rerank_score desc.
        """
        if not candidates:
            logger.warning("Reranker received empty candidate list.")
            return []

        # Build the passage list expected by FlashRank
        passages = [
            {"id": i, "text": sr.chunk.text, "meta": {"chunk_id": sr.chunk.chunk_id}}
            for i, sr in enumerate(candidates)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        # results is a list of dicts with keys: id, score, text, meta
        ranked: list[RankedResult] = []
        for res in results[: self._top_k]:
            original_idx = res["id"]
            ranked.append(
                RankedResult(
                    search_result=candidates[original_idx],
                    rerank_score=float(res["score"]),
                )
            )

        logger.debug(
            "Reranker kept %d/%d chunks. Top score: %.4f",
            len(ranked),
            len(candidates),
            ranked[0].rerank_score if ranked else 0.0,
        )
        return ranked