"""
retrieval.py - Hybrid Search combining Qdrant (vector) and BM25 (keyword).

Results from both backends are merged with Reciprocal Rank Fusion (RRF)
to produce a unified, re-ranked candidate list.

Qdrant API note: qdrant-client ≥ 1.7 removed the legacy `.search()` method.
This file uses `.query_points()` which is the current stable API.

Production fixes applied
------------------------
FIX 1 — Hash collisions: _chunk_id_to_int previously used abs(hash(...)) % 2**53
         which is NOT collision-resistant. Two different chunk_id strings can map
         to the same integer, silently overwriting a Qdrant point with no error.
         Replaced with uuid.uuid5() which is deterministic AND collision-resistant.

FIX 2 — Concurrent uploads destroying BM25: index_chunks() previously replaced
         self._bm25 and self._corpus_chunks entirely on each call. A second upload
         wiped everything from the first. Fixed by accumulating new chunks into the
         existing corpus (deduplicating by chunk_id) and rebuilding BM25 from the
         full accumulated list each time.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.ingestion import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A retrieved chunk with its combined RRF score."""
    chunk: Chunk
    score: float                         # RRF score (higher = better)
    vector_rank: Optional[int] = None    # Rank from Qdrant (1-based), None if absent
    bm25_rank: Optional[int] = None      # Rank from BM25 (1-based), None if absent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384        # all-MiniLM-L6-v2 output dimension
RRF_K           = 60         # Standard RRF constant (higher → smoother blending)


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------

class HybridSearch:
    """
    Hybrid search engine that blends dense vector search (Qdrant) with
    sparse keyword search (BM25) via Reciprocal Rank Fusion.

    Usage
    -----
    1. Instantiate once at startup.
    2. Call `index_chunks(chunks)` after each upload to add documents.
    3. Call `search(query, top_k)` to retrieve candidates.
    4. Call `_ensure_collection()` + reset `_bm25`/`_corpus_chunks` when
       clearing the store (done by clear_vector_store() in app.py).
    """

    def __init__(self, qdrant_path: str = "./qdrant_db"):
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Connecting to Qdrant (local disk mode) at '%s' …", qdrant_path)
        self._qdrant = QdrantClient(path=qdrant_path)

        # BM25 state – accumulated across all uploads in a session.
        # Reset to [] / None by clear_vector_store() in app.py.
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_chunks: list[Chunk] = []   # parallel list to BM25 index

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            logger.info("Creating Qdrant collection '%s'.", COLLECTION_NAME)
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        else:
            logger.info("Qdrant collection '%s' already exists.", COLLECTION_NAME)

    def collection_is_empty(self) -> bool:
        """Return True when the Qdrant collection holds no points."""
        info = self._qdrant.get_collection(COLLECTION_NAME)
        return (info.points_count or 0) == 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """
        Embed and upsert *chunks* into Qdrant, then rebuild the BM25 index.

        FIX 2 — Accumulation instead of replacement
        ─────────────────────────────────────────────
        Previously this method replaced self._corpus_chunks with the new
        batch, meaning a second upload wiped the BM25 index for the first.
        Now we:
          1. Identify which incoming chunk_ids are genuinely new.
          2. Extend self._corpus_chunks with only those new chunks.
          3. Rebuild BM25 from the full accumulated corpus.

        Qdrant already handles duplicates correctly via upsert semantics, so
        re-uploading the same file is safe — the vector points are overwritten
        in place and the BM25 corpus deduplication keeps one entry per chunk_id.
        """
        if not chunks:
            logger.warning("index_chunks called with empty chunk list – nothing to do.")
            return

        logger.info("Indexing %d chunks …", len(chunks))
        texts = [c.text for c in chunks]

        # --- Vector index (Qdrant) — unchanged, upsert is idempotent ---
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start: batch_start + batch_size]
            batch_texts  = texts[batch_start: batch_start + batch_size]

            embeddings = self._embedder.encode(batch_texts, show_progress_bar=False)

            points = [
                PointStruct(
                    # FIX 1 — uuid5 instead of abs(hash(...)) % 2**53
                    id=self._chunk_id_to_int(c.chunk_id),
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "source":   c.source,
                        "text":     c.text,
                        "metadata": c.metadata,
                    },
                )
                for c, emb in zip(batch_chunks, embeddings)
            ]

            self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.debug("Upserted batch of %d points.", len(points))

        # --- FIX 2 — Keyword index (BM25): accumulate, don't replace ---
        existing_ids: set[str] = {c.chunk_id for c in self._corpus_chunks}
        new_chunks   = [c for c in chunks if c.chunk_id not in existing_ids]

        if new_chunks:
            self._corpus_chunks.extend(new_chunks)
            logger.debug(
                "Added %d new chunks to BM25 corpus (total: %d).",
                len(new_chunks), len(self._corpus_chunks),
            )
        else:
            # All chunk_ids already present — this is a re-upload of the same file.
            # The Qdrant vectors are already updated above via upsert.
            # For BM25, replace the matching entries in-place so the text stays
            # consistent with the new Qdrant payloads.
            updated_map = {c.chunk_id: c for c in chunks}
            self._corpus_chunks = [
                updated_map.get(c.chunk_id, c) for c in self._corpus_chunks
            ]
            logger.debug(
                "Re-upload detected — updated %d existing BM25 corpus entries.",
                len(chunks),
            )

        # Rebuild BM25 from the full corpus (fast for typical doc sizes)
        tokenized_corpus = [c.text.lower().split() for c in self._corpus_chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            "Indexing complete. BM25 corpus: %d chunks total.",
            len(self._corpus_chunks),
        )

    def _load_bm25_from_qdrant(self) -> None:
        """
        Reconstruct the BM25 index by scrolling all payloads from Qdrant.
        Called automatically on the first search after a process restart.
        """
        logger.info("Rebuilding BM25 index from Qdrant payloads …")
        chunks: list[Chunk] = []
        offset = None

        while True:
            results, offset = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                p = point.payload
                chunks.append(
                    Chunk(
                        chunk_id=p["chunk_id"],
                        source=p["source"],
                        text=p["text"],
                        metadata=p.get("metadata", {}),
                    )
                )
            if offset is None:
                break

        if chunks:
            tokenized = [c.text.lower().split() for c in chunks]
            self._bm25 = BM25Okapi(tokenized)
            self._corpus_chunks = chunks
            logger.info("BM25 index rebuilt from %d stored chunks.", len(chunks))
        else:
            logger.warning("Qdrant collection is empty; BM25 index not built.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 15) -> list[SearchResult]:
        """
        Execute hybrid search and return up to *top_k* fully-populated
        SearchResult objects.

        Steps:
          1. Embed query → dense search via Qdrant `.query_points()`.
          2. Tokenise query → BM25 keyword search over in-memory corpus.
          3. Merge both ranked lists with Reciprocal Rank Fusion (RRF).
          4. Hydrate stub results with real Chunk text from the corpus map.
        """
        # Lazily rebuild BM25 after a process restart
        if self._bm25 is None:
            self._load_bm25_from_qdrant()

        query_vector = self._embedder.encode(query).tolist()

        vector_hits = self._vector_search(query_vector, top_k=top_k)
        bm25_hits   = self._bm25_search(query, top_k=top_k)

        raw_results = self._reciprocal_rank_fusion(vector_hits, bm25_hits, top_k=top_k)

        # Hydrate: resolve stub chunk_ids to full Chunk objects
        corpus_map: dict[str, Chunk] = {c.chunk_id: c for c in self._corpus_chunks}

        hydrated: list[SearchResult] = []
        for r in raw_results:
            full_chunk = corpus_map.get(r.chunk.chunk_id)
            if full_chunk is None:
                logger.warning("Could not resolve chunk '%s' – skipping.", r.chunk.chunk_id)
                continue
            hydrated.append(
                SearchResult(
                    chunk=full_chunk,
                    score=r.score,
                    vector_rank=r.vector_rank,
                    bm25_rank=r.bm25_rank,
                )
            )

        return hydrated

    # ------------------------------------------------------------------
    # Private search backends
    # ------------------------------------------------------------------

    def _vector_search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Dense ANN search via Qdrant's current `.query_points()` API.
        Returns (chunk_id, score) tuples ordered by cosine similarity desc.
        """
        response = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [(pt.payload["chunk_id"], pt.score) for pt in response.points]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Sparse keyword search via BM25Okapi.
        Returns (chunk_id, bm25_score) tuples ordered by score desc.
        """
        if self._bm25 is None or not self._corpus_chunks:
            logger.warning("BM25 index unavailable; returning empty results.")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored = sorted(
            zip([c.chunk_id for c in self._corpus_chunks], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_hits: list[tuple[str, float]],
        bm25_hits:   list[tuple[str, float]],
        top_k: int,
        k: int = RRF_K,
    ) -> list[SearchResult]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))

        Returns stub SearchResult objects (chunk.text is empty string);
        the caller is responsible for hydrating them with real Chunk data.
        """
        vector_rank: dict[str, int] = {
            cid: rank + 1 for rank, (cid, _) in enumerate(vector_hits)
        }
        bm25_rank: dict[str, int] = {
            cid: rank + 1 for rank, (cid, _) in enumerate(bm25_hits)
        }

        all_ids = set(vector_rank) | set(bm25_rank)

        rrf_scores: dict[str, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in vector_rank:
                score += 1.0 / (k + vector_rank[cid])
            if cid in bm25_rank:
                score += 1.0 / (k + bm25_rank[cid])
            rrf_scores[cid] = score

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        ranked_ids = ranked_ids[:top_k]

        results: list[SearchResult] = []
        for cid in ranked_ids:
            stub = Chunk(chunk_id=cid, source="", text="")
            results.append(
                SearchResult(
                    chunk=stub,
                    score=rrf_scores[cid],
                    vector_rank=vector_rank.get(cid),
                    bm25_rank=bm25_rank.get(cid),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        """
        FIX 1 — Collision-resistant chunk_id → Qdrant point ID mapping.

        Old implementation:
            abs(hash(chunk_id)) % (2 ** 53)
        Problem: Python's hash() is NOT collision-resistant. Two different
        strings can produce the same integer, causing Qdrant to silently
        overwrite a point with no error or warning. With 10 000+ chunks
        the birthday problem makes collisions likely.

        New implementation:
            uuid.uuid5(NAMESPACE_DNS, chunk_id)
        uuid5 is a deterministic SHA-1-based UUID. For any given input
        string it always produces the same UUID, and the collision
        probability over the full 128-bit space is astronomically small.
        We take the lower 63 bits (>> 65) to stay within Qdrant's safe
        unsigned integer range while preserving full determinism.
        """
        return uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).int >> 65