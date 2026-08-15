"""Fase 1.3 — el mismo pipeline de RAG, envuelto en una API REST."""

import logging
import time

from anthropic import APIConnectionError, APIError, APIStatusError, RateLimitError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.rag import answer, ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")

app = FastAPI(title="RAG System")


class QueryRequest(BaseModel):
    question: str


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="solo se aceptan archivos PDF")

    pdf_bytes = await file.read()
    start = time.monotonic()
    try:
        result = ingest_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="límite de tasa del proveedor de embeddings alcanzado")
    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("error del proveedor durante ingesta: %s", e)
        raise HTTPException(status_code=502, detail="error del proveedor de embeddings")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "ingest chunks=%d tokens=%d cost_usd=%.6f latency_ms=%.0f",
        result["chunks"], result["embedding_tokens"], result["cost_usd"], latency_ms,
    )
    return result


@app.post("/query")
def query(request: QueryRequest):
    start = time.monotonic()
    try:
        result = answer(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        logger.error("respuesta vacía del proveedor: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="límite de tasa del proveedor de LLM alcanzado")
    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("error del proveedor durante query: %s", e)
        raise HTTPException(status_code=502, detail="error del proveedor de LLM")

    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "query question=%r n_sources=%d cost_usd=%.6f latency_ms=%.0f",
        request.question[:80], len(result["sources"]), result["cost_usd"], latency_ms,
    )
    return result
