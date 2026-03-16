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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("app")

BASE_DIR     = Path(__file__).parent
UPLOAD_DIR   = BASE_DIR / "uploads"
SESSIONS_DIR = BASE_DIR / "sessions"
QDRANT_PATH  = str(BASE_DIR / "qdrant_db")

SESSION_LOCK_FILE = BASE_DIR / "indexed_session.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

MAX_CHARS_PER_CHUNK = 1_500  # ≈ 375 tokens at 4 chars/token

MAX_QUERY_CHARS = 2_000

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown"}

UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Chat Interface", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Mount static files after the FastAPI app is created
app.mount("/static", StaticFiles(directory="static"), name="static")

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

    engine._bm25 = None
    engine._corpus_chunks = []

    set_indexed_session_id(None)

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



@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    user_input: str = Form(...),
    session_id: str = Form(...),
):
    user_input = user_input.strip()[:MAX_QUERY_CHARS]
    if not user_input:
        return HTMLResponse("", status_code=204)

    engine = get_search_engine()

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

@app.post("/session/new")
async def new_session(request: Request):
    await clear_vector_store()
    new_id = str(uuid.uuid4())[:8]
    logger.info("New session created: %s", new_id)
    return JSONResponse({"session_id": new_id})
