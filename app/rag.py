"""Pipeline de RAG compartido por los endpoints: ingestión y query.

Misma lógica de exercises/03_rag_ingest.py y exercises/04_rag_query.py,
reutilizada acá para no reimplementar el pipeline dos veces.
"""

import os
import re
from io import BytesIO

import psycopg
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

VOYAGE_PRICE_PER_M = 0.06
DEEPSEEK_INPUT_PRICE_PER_M = 0.14
DEEPSEEK_OUTPUT_PRICE_PER_M = 0.28

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
llm = Anthropic(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
)


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    raw = "\n".join(page.extract_text() for page in reader.pages)
    return re.sub(r"\s+", " ", raw).strip()


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text: str, size: int = 500) -> list[str]:
    sentences = split_sentences(text)
    chunks = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > size:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_pdf(pdf_bytes: bytes) -> dict:
    text = extract_text(pdf_bytes)
    if not text:
        raise ValueError("no se pudo extraer texto del PDF")

    chunks = chunk_text(text)
    result = voyage.embed(chunks, model="voyage-3.5", input_type="document")
    cost = (result.total_tokens / 1_000_000) * VOYAGE_PRICE_PER_M

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR(1024)
            )
        """)
        conn.execute("TRUNCATE chunks")
        for content, embedding in zip(chunks, result.embeddings):
            conn.execute(
                "INSERT INTO chunks (content, embedding) VALUES (%s, %s)",
                (content, str(embedding)),
            )

    return {
        "chunks": len(chunks),
        "embedding_tokens": result.total_tokens,
        "cost_usd": round(cost, 6),
    }


def retrieve(question: str, k: int = 3) -> tuple[list[dict], int]:
    result = voyage.embed([question], model="voyage-3.5", input_type="query")
    query_embedding = result.embeddings[0]

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """
            SELECT content, embedding <=> %s AS distance
            FROM chunks
            ORDER BY distance
            LIMIT %s
            """,
            (str(query_embedding), k),
        ).fetchall()

    sources = [{"content": content, "distance": float(distance)} for content, distance in rows]
    return sources, result.total_tokens


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Respondé la pregunta usando SOLO la información del contexto.
Si el contexto no alcanza para responder, decilo explícitamente.

Contexto:
{context}

Pregunta: {question}"""


def answer(question: str, k: int = 3) -> dict:
    if not question.strip():
        raise ValueError("la pregunta no puede estar vacía")

    sources, embedding_tokens = retrieve(question, k)
    if not sources:
        raise LookupError("no hay documentos ingeridos todavía")

    prompt = build_prompt(question, [s["content"] for s in sources])
    message = llm.messages.create(
        model="deepseek-v4-flash",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost = (
        (embedding_tokens / 1_000_000) * VOYAGE_PRICE_PER_M
        + (input_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_M
        + (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_M
    )
    answer_text = "".join(block.text for block in message.content if block.type == "text")
    if not answer_text.strip():
        raise RuntimeError(
            f"el modelo no devolvió texto (stop_reason={message.stop_reason}); "
            "probablemente se quedó sin max_tokens mientras pensaba"
        )

    return {
        "answer": answer_text,
        "sources": sources,
        "usage": {
            "embedding_tokens": embedding_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "cost_usd": round(cost, 6),
    }
