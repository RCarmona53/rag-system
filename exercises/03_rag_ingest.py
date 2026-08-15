"""Fase 1.2 — RAG manual, sin framework: ingestión.

PDF -> texto -> chunks por oraciones completas -> embeddings -> pgvector.

Nota: este PDF en particular no tiene saltos de línea en blanco reales entre
párrafos (pypdf pone casi cada palabra en su propia línea), así que "chunking
por párrafo" no es distinguible de "chunking por palabra". Por eso agrupamos
por oraciones completas en vez de por líneas en blanco.
"""

import os
import re

import psycopg
import voyageai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
PDF_PATH = "/home/rodrigo/Descargas/Resume-Rodrigo-Carmona.pdf"

CHUNK_SIZE = 500

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    raw = "\n".join(page.extract_text() for page in reader.pages)
    return re.sub(r"\s+", " ", raw).strip()


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_text(text: str, size: int) -> list[str]:
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


def main() -> None:
    text = extract_text(PDF_PATH)
    chunks = chunk_text(text, CHUNK_SIZE)
    print(f"texto extraído: {len(text)} caracteres -> {len(chunks)} chunks")

    result = voyage.embed(chunks, model="voyage-3.5", input_type="document")

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

    print(f"{len(chunks)} chunks insertados en pgvector.")


if __name__ == "__main__":
    main()
