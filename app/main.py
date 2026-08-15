"""Fase 1.3 — el mismo pipeline de RAG, envuelto en una API REST.

Proyecto 2 agrega `POST /agent` (agente con tool-calling, `app/agent.py`) de
forma puramente aditiva: `/query` y `/documents` de abajo no se tocan.
"""

import logging
import time
from contextlib import asynccontextmanager

from anthropic import APIConnectionError, APIError, APIStatusError, RateLimitError
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent import GraphRecursionError, run_agent
from app.agent import setup_checkpointer as agent_setup_checkpointer
from app.agent import teardown_checkpointer as agent_teardown_checkpointer
from app.rag import answer, ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_setup_checkpointer()
    yield
    agent_teardown_checkpointer()


app = FastAPI(title="RAG System", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class AgentRequest(BaseModel):
    question: str
    thread_id: str | None = None


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
    latency_ms: float
    cost_usd: float
    thread_id: str


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


@app.post("/agent")
def agent_query(request: AgentRequest):
    try:
        result = run_agent(request.question, thread_id=request.thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RateLimitError:
        raise HTTPException(status_code=429, detail="límite de tasa del proveedor de LLM alcanzado")
    except GraphRecursionError:
        raise HTTPException(status_code=503, detail="el agente superó el límite de recursión del grafo")
    except (APIConnectionError, APIStatusError, APIError) as e:
        logger.error("error del proveedor durante agent: %s", e)
        raise HTTPException(status_code=502, detail="error del proveedor de LLM")

    logger.info(
        "agent question=%r tools_used=%s cost_usd=%.6f latency_ms=%.0f thread_id=%s",
        request.question[:80], result["tools_used"], result["cost_usd"], result["latency_ms"], result["thread_id"],
    )
    return result
