# Local-First RAG System

A **production-ready Retrieval-Augmented Generation** pipeline built from scratch — no LangChain, no LlamaIndex.

| Layer | Technology |
|---|---|
| PDF parsing | `pdfplumber` |
| Markdown chunking | custom heading-aware splitter |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally) |
| Vector store | `qdrant-client` (local disk mode, no server needed) |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Hybrid merge | Reciprocal Rank Fusion (pure Python) |
| Reranker | `flashrank` (local cross-encoder) |
| LLM generation | Hugging Face Inference API via `huggingface-hub` |
| Memory | JSON-backed `ChatSession` (last 3 turns injected into prompt) |

---

## Directory Structure

```
rag_system/
├── main.py                  ← Entry point (ingest / query / chat modes)
├── requirements.txt
├── README.md
├── docs/                    ← Drop your PDF / Markdown files here
├── qdrant_db/               ← Auto-created by Qdrant (local disk mode)
├── sessions/                ← Auto-created; one JSON file per chat session
└── src/
    ├── __init__.py
    ├── ingestion.py         ← PDF + Markdown parsing & chunking
    ├── retrieval.py         ← HybridSearch (Qdrant + BM25 + RRF)
    ├── reranker.py          ← FlashRank cross-encoder reranking
    ├── memory.py            ← ChatSession with JSON persistence
    └── generator.py         ← HF InferenceClient + exponential back-off
```

---

## Prerequisites

- Python 3.10+
- A [Hugging Face account](https://huggingface.co/) with an API token that has *Inference* access

---

## Installation

```bash
# 1. Clone / download the project
cd rag_system

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Offline embeddings note:** `sentence-transformers` will download `all-MiniLM-L6-v2`
> the first time it runs (≈ 90 MB). After that it works fully offline.

---

## Configuration

All settings can be overridden via **environment variables** before running:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *required* | Your Hugging Face API token |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HF model ID to use for generation |
| `QDRANT_PATH` | `./qdrant_db` | Local path for Qdrant on-disk storage |
| `DOCS_DIR` | `./docs` | Default documents directory |
| `SESSIONS_DIR` | `./sessions` | Directory for session JSON files |

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

---

## Quick Start

### Step 1 — Add your documents

Copy 10–20 PDF and/or Markdown files into the `docs/` directory:

```bash
mkdir -p docs
cp /path/to/your/*.pdf docs/
cp /path/to/your/*.md  docs/
```

### Step 2 — Initialize / ingest the Qdrant collection

This parses every document, embeds the chunks, and populates the vector store.
**Run this once** (or re-run whenever your document set changes):

```bash
python main.py --mode ingest --docs ./docs
```

What happens under the hood:
1. `ingestion.py` parses each file with `pdfplumber` (PDF) or heading-aware splitting (Markdown).
2. Chunks are embedded locally with `all-MiniLM-L6-v2`.
3. Embeddings are upserted into a `qdrant_db/` directory (no server process needed).
4. A BM25 index is built from the same corpus and held in memory.

### Step 3a — Interactive chat

```bash
python main.py --mode chat --session my_project
```

Commands inside the chat loop:

| Input | Action |
|---|---|
| Any question | Runs the full RAG pipeline |
| `clear` | Wipes conversation history for the session |
| `exit` / `quit` | Exits |

### Step 3b — Single question (non-interactive)

```bash
python main.py --mode query \
    --query "How does the attention mechanism work?" \
    --session my_project
```

---

## Pipeline Deep Dive

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  HybridSearch.search(query, top_k=15)       │
│                                             │
│  ┌──────────────┐   ┌──────────────────┐    │
│  │ Qdrant dense │   │  BM25 keyword    │    │
│  │ (cosine sim) │   │  (Okapi BM25)    │    │
│  └──────┬───────┘   └────────┬─────────┘    │
│         │                    │              │
│         └────────┬───────────┘              │
│                  ▼                          │
│       Reciprocal Rank Fusion (RRF)          │
│       score = Σ  1/(60 + rank_r)            │
└──────────────────┬──────────────────────────┘
                   │ top-15 candidates
                   ▼
┌─────────────────────────────────────────────┐
│  ChunkReranker.rerank(query, candidates)    │
│  FlashRank cross-encoder → top-5 passages   │
└──────────────────┬──────────────────────────┘
                   │ 5 grounding passages
                   ▼
┌─────────────────────────────────────────────┐
│  RAGGenerator.generate(query, context,      │
│                        history)             │
│                                             │
│  Prompt = system + context + last 3 turns   │
│         + current question                  │
│                                             │
│  HF InferenceClient.chat_completion()       │
│  Retry on 429 / 503 with exponential        │
│  back-off (up to 5 attempts)                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
              Answer (text)
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  ChatSession.add_turn(user/assistant)       │
│  Persisted to sessions/<id>.json            │
└─────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why RRF instead of a weighted sum?

RRF is **score-scale agnostic** — it uses ranks, not raw scores. Qdrant cosine similarities (0–1) and BM25 scores (0–∞) live on incompatible scales; converting ranks first avoids the need to tune weighting coefficients.

### Why FlashRank?

It runs a full cross-encoder *locally* with no API call, adding only ~50ms latency. It re-scores the 15-candidate shortlist with the actual query, catching cases where keyword or semantic similarity was misleading.

### Why a strict grounding prompt?

The system prompt explicitly forbids the model from using outside knowledge. This makes the system behave as a **document Q&A tool**, not a general chatbot, preventing hallucinations in technical domains.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HF_TOKEN` not set | `export HF_TOKEN="hf_..."` before running |
| 503 errors from HF | The model is cold-starting; the retry loop handles up to 5 attempts automatically |
| Empty retrieval results | Verify that ingestion completed (`qdrant_db/` directory must exist and be non-empty) |
| PDF extraction warnings | Normal for scanned PDFs; consider OCR pre-processing for image-only PDFs |
| BM25 index not loaded | Restart is handled automatically — BM25 is rebuilt from Qdrant payloads on first search |

---

## License

MIT