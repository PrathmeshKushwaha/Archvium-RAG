"""
main.py - Entry point for the local-first RAG system.

Usage
-----
    # Step 1: Ingest documents (run once, or whenever docs change)
    python main.py --mode ingest --docs ./docs

    # Step 2: Start an interactive chat session
    python main.py --mode chat --session my_session

    # Step 3: Ask a single question non-interactively
    python main.py --mode query --query "What is the attention mechanism?"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.ingestion import ingest_documents
from src.retrieval import HybridSearch
from src.reranker import ChunkReranker
from src.memory import ChatSession
from src.generator import RAGGenerator, GenerationConfig

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Configuration (override via environment variables or CLI flags)
# ---------------------------------------------------------------------------

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./sessions")
HF_TOKEN = os.getenv("HF_TOKEN")                            # Required for generation
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

RETRIEVAL_TOP_K = 15   # Candidates passed to the reranker
RERANKER_TOP_K = 5     # Final passages injected into the prompt
HISTORY_TURNS = 3      # Number of past exchanges included in the prompt


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_components(session_id: str = "default"):
    """Instantiate and return all pipeline components."""
    search_engine = HybridSearch(qdrant_path=QDRANT_PATH)
    reranker = ChunkReranker(top_k=RERANKER_TOP_K)
    memory = ChatSession(
        session_id=session_id,
        storage_dir=SESSIONS_DIR,
        history_turns=HISTORY_TURNS,
    )
    generator = RAGGenerator(
        model_id=HF_MODEL,
        hf_token=HF_TOKEN,
        gen_config=GenerationConfig(max_new_tokens=512, temperature=0.1),
    )
    return search_engine, reranker, memory, generator


def answer_query(
    query: str,
    search_engine: HybridSearch,
    reranker: ChunkReranker,
    memory: ChatSession,
    generator: RAGGenerator,
) -> str:
    """
    Full RAG pipeline for a single query:
      1. Hybrid search (vector + BM25, merged via RRF)
      2. Reranking (top-15 → top-5)
      3. Prompt construction with conversation history
      4. LLM generation (HF Inference API with back-off)
      5. Memory persistence
    """
    logger.info("Query: %s", query)

    # --- Retrieval ---
    candidates = search_engine.search(query, top_k=RETRIEVAL_TOP_K)
    logger.info("Retrieved %d candidates.", len(candidates))

    if not candidates:
        answer = "I could not find any relevant passages in the document collection."
        memory.add_turn("user", query)
        memory.add_turn("assistant", answer)
        return answer

    # --- Reranking ---
    ranked = reranker.rerank(query, candidates)
    logger.info("Reranker selected %d passages.", len(ranked))

    context_texts = [r.search_result.chunk.text for r in ranked]

    # --- History ---
    history = memory.get_history_for_prompt()

    # --- Generation ---
    logger.info("Calling HF Inference API …")
    answer = generator.generate(query, context_texts, history=history)

    # --- Persist turn ---
    memory.add_turn("user", query)
    memory.add_turn("assistant", answer)

    return answer


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def mode_ingest(docs_dir: str) -> None:
    """Parse documents and populate the Qdrant + BM25 index."""
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        logger.error("Documents directory '%s' does not exist.", docs_dir)
        sys.exit(1)

    logger.info("Starting ingestion from '%s' …", docs_dir)
    chunks = ingest_documents(docs_path)

    if not chunks:
        logger.error("No chunks produced – check that '%s' contains PDF or Markdown files.", docs_dir)
        sys.exit(1)

    search_engine = HybridSearch(qdrant_path=QDRANT_PATH)
    search_engine.index_chunks(chunks)
    logger.info("Ingestion complete. %d chunks indexed.", len(chunks))


def mode_query(query: str, session_id: str = "default") -> None:
    """Answer a single question and print the result."""
    search_engine, reranker, memory, generator = build_components(session_id)

    if search_engine.collection_is_empty():
        logger.error(
            "Qdrant collection is empty. Run `python main.py --mode ingest` first."
        )
        sys.exit(1)

    answer = answer_query(query, search_engine, reranker, memory, generator)
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(answer)
    print("=" * 70 + "\n")


def mode_chat(session_id: str = "default") -> None:
    """Interactive REPL chat loop."""
    search_engine, reranker, memory, generator = build_components(session_id)

    if search_engine.collection_is_empty():
        logger.error(
            "Qdrant collection is empty. Run `python main.py --mode ingest` first."
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" RAG Chat System  |  Session:", memory.session_id)
    print(" Type 'exit' or 'quit' to stop.  Type 'clear' to reset history.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            memory.clear()
            print("History cleared.\n")
            continue

        try:
            answer = answer_query(user_input, search_engine, reranker, memory, generator)
            print(f"\nAssistant: {answer}\n")
        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline error: %s", exc, exc_info=True)
            print(f"[Error] {exc}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-first RAG system with Hybrid Search and HF generation."
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "chat"],
        default="chat",
        help="Operation mode (default: chat).",
    )
    parser.add_argument(
        "--docs",
        default=DOCS_DIR,
        help="Directory containing PDF/Markdown documents (used in 'ingest' mode).",
    )
    parser.add_argument(
        "--query",
        help="Question to answer (used in 'query' mode).",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session ID for conversation memory (default: 'default').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "ingest":
        mode_ingest(args.docs)

    elif args.mode == "query":
        if not args.query:
            logger.error("'query' mode requires --query <question>.")
            sys.exit(1)
        mode_query(args.query, session_id=args.session)

    elif args.mode == "chat":
        mode_chat(session_id=args.session)