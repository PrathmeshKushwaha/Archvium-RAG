import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
load_dotenv()

# ── RAG pipeline imports ────────────────────────────────────────────────────
from src.ingestion import ingest_documents, Chunk
from src.retrieval import HybridSearch, COLLECTION_NAME
from src.reranker import ChunkReranker
from src.memory import ChatSession
from src.generator import RAGGenerator, GenerationConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Directories & constants
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
SESSIONS_DIR = BASE_DIR / "sessions"
QDRANT_PATH  = str(BASE_DIR / "qdrant_db")

# FIX 3 — shared state file readable by all uvicorn worker processes
SESSION_LOCK_FILE = BASE_DIR / "indexed_session.json"

# FIX 4a — reject uploads larger than this before touching disk
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# FIX 6 — prevent context window overflow by capping each chunk's char length
MAX_CHARS_PER_CHUNK = 1_500  # ≈ 375 tokens at 4 chars/token

# FIX 10 — hard cap on user query length before any processing
MAX_QUERY_CHARS = 2_000

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown"}

UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Chat Interface", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Mount static files after the FastAPI app is created
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# RAG component singletons
# ---------------------------------------------------------------------------
_search_engine: HybridSearch | None = None
_reranker: ChunkReranker | None = None
_generator: RAGGenerator | None = None


def get_search_engine() -> HybridSearch:
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch(qdrant_path=QDRANT_PATH)
    return _search_engine


def get_reranker() -> ChunkReranker:
    global _reranker
    if _reranker is None:
        _reranker = ChunkReranker(top_k=5)
    return _reranker


def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator(
            gen_config=GenerationConfig(max_new_tokens=512, temperature=0.1)
        )
    return _generator


def get_session(session_id: str) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        storage_dir=str(SESSIONS_DIR),
        history_turns=3,
    )


# ---------------------------------------------------------------------------
# FIX 3 — Session lock file helpers
#
# Replaces the old in-process `_indexed_session_id` variable.
# All uvicorn workers share the same filesystem, so writing/reading a small
# JSON file gives consistent session ownership across processes.
# For a true multi-user production system, swap this for Redis.
# ---------------------------------------------------------------------------

def get_indexed_session_id() -> str | None:
    """Return the session ID that currently owns the vector index, or None."""
    try:
        if SESSION_LOCK_FILE.exists():
            data = json.loads(SESSION_LOCK_FILE.read_text(encoding="utf-8"))
            return data.get("session_id")
    except Exception as exc:
        logger.warning("Could not read session lock file: %s", exc)
    return None


def set_indexed_session_id(session_id: str | None) -> None:
    """Persist the owning session ID to disk (None clears it)."""
    try:
        if session_id is None:
            SESSION_LOCK_FILE.unlink(missing_ok=True)
        else:
            SESSION_LOCK_FILE.write_text(
                json.dumps({"session_id": session_id}), encoding="utf-8"
            )
    except Exception as exc:
        logger.error("Could not write session lock file: %s", exc)


# ---------------------------------------------------------------------------
# Vector store lifecycle
# ---------------------------------------------------------------------------

def _do_clear_vector_store() -> None:
    """
    Synchronous core of clear_vector_store — safe to run in a thread.

    - Deletes and recreates the Qdrant collection.
    - Resets BM25 in-memory state on the singleton.
    - Clears the session lock file (FIX 3).
    - Wipes the uploads directory (FIX 4b — prevents disk fill).
    """
    engine = get_search_engine()

    # Wipe Qdrant collection
    try:
        engine._qdrant.delete_collection(COLLECTION_NAME)
        logger.info("Qdrant collection '%s' deleted.", COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Could not delete collection (may not exist): %s", exc)

    engine._ensure_collection()

    # Reset BM25 state (FIX 2 accumulation resets cleanly here)
    engine._bm25 = None
    engine._corpus_chunks = []

    # FIX 3 — clear shared lock file
    set_indexed_session_id(None)

    # FIX 4b — wipe uploaded files so disk doesn't fill up
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Upload directory cleared.")
    except Exception as exc:
        logger.error("Failed to clear upload directory: %s", exc)

    logger.info("Vector store cleared and recreated.")


async def clear_vector_store() -> None:
    """
    Async wrapper — runs the blocking clear in a thread (FIX 5).
    Call this from route handlers with `await`.
    """
    await asyncio.to_thread(_do_clear_vector_store)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the main chat UI with a fresh session ID.
    Clears the vector store so a page refresh never serves stale documents.
    """
    session_id = str(uuid.uuid4())[:8]
    await clear_vector_store()
    logger.info("New page load — session %s, vector store cleared.", session_id)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "session_id": session_id},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.time(),
        "indexed_session": get_indexed_session_id(),
    }


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Validate, save, and ingest an uploaded document.

    Fixes applied:
      FIX 4a — rejects files over MAX_UPLOAD_BYTES before writing to disk.
      FIX 5  — CPU-bound ingest and embed run in asyncio.to_thread().
      FIX 3  — session ownership written to disk lock file after indexing.
    """
    suffix = Path(file.filename).suffix.lower()

    # ── Validate file type ──────────────────────────────────────────────────
    if suffix not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "components/upload_status.html",
            {
                "request": request,
                "success": False,
                "filename": file.filename,
                "message": (
                    f"Unsupported file type '{suffix}'. "
                    "Please upload a PDF or Markdown file."
                ),
            },
            status_code=415,
        )

    # ── FIX 4a — read into memory and enforce size limit ───────────────────
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        return templates.TemplateResponse(
            "components/upload_status.html",
            {
                "request": request,
                "success": False,
                "filename": file.filename,
                "message": "Failed to read the uploaded file. Please try again.",
            },
            status_code=500,
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return templates.TemplateResponse(
            "components/upload_status.html",
            {
                "request": request,
                "success": False,
                "filename": file.filename,
                "message": (
                    f"File is too large ({len(contents) // (1024*1024)} MB). "
                    f"Maximum allowed size is {limit_mb} MB."
                ),
            },
            status_code=413,
        )

    # ── Save to session-scoped directory ────────────────────────────────────
    session_upload_dir = UPLOAD_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    dest = session_upload_dir / file.filename

    try:
        dest.write_bytes(contents)
        logger.info("Saved upload: %s (%d bytes)", dest, len(contents))
    except Exception as exc:
        logger.error("Failed to save upload: %s", exc)
        return templates.TemplateResponse(
            "components/upload_status.html",
            {
                "request": request,
                "success": False,
                "filename": file.filename,
                "message": "Server error while saving the file. Please try again.",
            },
            status_code=500,
        )

    # ── FIX 5 — run CPU-bound ingestion in a thread ─────────────────────────
    try:
        chunks: list[Chunk] = await asyncio.to_thread(
            ingest_documents, session_upload_dir
        )
        if not chunks:
            raise ValueError(
                "No text could be extracted from the document. "
                "If this is a scanned PDF, OCR support may be required."
            )

        engine = get_search_engine()
        await asyncio.to_thread(engine.index_chunks, chunks)

        # FIX 3 — persist session ownership to the shared lock file
        set_indexed_session_id(session_id)

        logger.info(
            "Ingested %d chunks from '%s' for session '%s'.",
            len(chunks), file.filename, session_id,
        )

    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", file.filename, exc)
        return templates.TemplateResponse(
            "components/upload_status.html",
            {
                "request": request,
                "success": False,
                "filename": file.filename,
                "message": f"Ingestion failed: {exc}",
            },
            status_code=422,
        )

    return templates.TemplateResponse(
        "components/upload_status.html",
        {
            "request": request,
            "success": True,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "message": f"Document ready — {len(chunks)} passages indexed.",
        },
    )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    user_input: str = Form(...),
    session_id: str = Form(...),
):
    """
    Run the full RAG pipeline and return ONLY the AI response HTML fragment.

    Fixes applied:
      FIX 10 — query hard-capped at MAX_QUERY_CHARS before any processing.
      FIX 3  — session ownership checked via disk lock file, not in-process var.
      FIX 6  — context chunks truncated to MAX_CHARS_PER_CHUNK before generation.
      FIX 5  — search, rerank, and generation run in asyncio.to_thread().
    """
    # FIX 10 — cap query length before touching anything else
    user_input = user_input.strip()[:MAX_QUERY_CHARS]
    if not user_input:
        return HTMLResponse("", status_code=204)

    engine = get_search_engine()

    # ── FIX 3 — check session ownership via shared lock file ────────────────
    # Guards both "collection empty" and "belongs to a different session"
    # (the latter catches the page-refresh stale-data case across workers).
    no_docs = (
        engine.collection_is_empty()
        or get_indexed_session_id() != session_id
    )
    if no_docs:
        return templates.TemplateResponse(
            "components/ai_message.html",
            {
                "request": request,
                "content": (
                    "⚠️ No documents have been uploaded in this session. "
                    "Please upload a PDF or Markdown file using the sidebar "
                    "before asking questions."
                ),
                "is_error": True,
            },
        )

    # ── FIX 5 — run pipeline steps in threads ───────────────────────────────
    try:
        reranker  = get_reranker()
        generator = get_generator()
        memory    = get_session(session_id)

        # Retrieval (CPU-bound: embedding + Qdrant ANN + BM25)
        candidates = await asyncio.to_thread(engine.search, user_input, 15)

        # Reranking (CPU-bound: cross-encoder inference)
        ranked = await asyncio.to_thread(reranker.rerank, user_input, candidates)

        # FIX 6 — truncate each chunk to avoid context window overflow
        context = [
            r.search_result.chunk.text[:MAX_CHARS_PER_CHUNK]
            for r in ranked
        ]

        history = memory.get_history_for_prompt()

        # Generation (network-bound but blocks on synchronous HTTP client)
        answer = await asyncio.to_thread(
            generator.generate, user_input, context, history
        )

        # Persist turn (FIX 7 turn cap is enforced inside ChatSession.add_turn)
        memory.add_turn("user", user_input)
        memory.add_turn("assistant", answer)

        logger.info(
            "Response generated for session '%s' (%d chars).",
            session_id, len(answer),
        )

    except Exception as exc:
        logger.error(
            "Pipeline error for session '%s': %s", session_id, exc, exc_info=True
        )
        return templates.TemplateResponse(
            "components/ai_message.html",
            {
                "request": request,
                "content": f"An error occurred while generating a response: {exc}",
                "is_error": True,
            },
            status_code=500,
        )

    return templates.TemplateResponse(
        "components/ai_message.html",
        {
            "request": request,
            "content": answer,
            "is_error": False,
        },
    )


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

@app.post("/session/new")
async def new_session(request: Request):
    """
    Clear all server-side state and return a fresh session ID as JSON.

    FIX 3 — clear_vector_store() now writes/clears the disk lock file
             instead of the old in-process variable, so all workers
             see the reset immediately.
    FIX 4b — upload directory is wiped inside clear_vector_store().
    FIX 5  — vector store clearing runs in a thread.
    """
    await clear_vector_store()
    new_id = str(uuid.uuid4())[:8]
    logger.info("New session created: %s", new_id)
    return JSONResponse({"session_id": new_id})