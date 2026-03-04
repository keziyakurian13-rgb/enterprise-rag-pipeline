"""
Enterprise RAG Pipeline - FastAPI Server
Production-ready REST API for document ingestion and Q&A
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import time
import logging
from pathlib import Path

from rag import EnterpriseRAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise RAG Q&A API",
    description="Production-grade Retrieval-Augmented Generation pipeline using ChromaDB + LangChain + HuggingFace",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_pipeline = EnterpriseRAGPipeline(use_openai=bool(os.getenv("OPENAI_API_KEY")))

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- Request / Response Models ----------

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    num_sources: int
    latency_ms: float


class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    chunks_created: int
    vectorstore_size: int
    latency_ms: float


# ---------- Endpoints ----------

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Enterprise RAG Pipeline",
        "status": "running",
        "version": "1.0.0",
        "tech": "ChromaDB + LangChain + HuggingFace + FastAPI",
    }


@app.get("/health", tags=["Health"])
async def health():
    stats = rag_pipeline.get_stats()
    return {"status": "healthy", "vectorstore": stats}


@app.post("/ingest/upload", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF or TXT document into the vector store.
    Documents are chunked with sliding window strategy (512 tokens, 64 overlap).
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    start = time.time()
    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = rag_pipeline.ingest_documents(str(file_path))
        latency = round((time.time() - start) * 1000, 2)

        return IngestResponse(
            success=True,
            message=f"Successfully ingested '{file.filename}'",
            latency_ms=latency,
            **result,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion error: {str(e)}")


@app.post("/ingest/path", tags=["Ingestion"])
async def ingest_path(path: str):
    """Ingest documents from a server-side file path or directory."""
    start = time.time()
    try:
        result = rag_pipeline.ingest_documents(path)
        latency = round((time.time() - start) * 1000, 2)
        return {"success": True, "latency_ms": latency, **result}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query(request: QueryRequest):
    """
    Semantic search + LLM inference.
    Uses MMR retrieval for diverse, context-rich answers.
    """
    start = time.time()
    try:
        result = rag_pipeline.query(request.question)
        latency = round((time.time() - start) * 1000, 2)
        return QueryResponse(latency_ms=latency, **result)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/stats", tags=["Monitoring"])
async def stats():
    """Return vectorstore statistics and pipeline configuration."""
    return rag_pipeline.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
