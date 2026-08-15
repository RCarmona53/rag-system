"""Fase 1.2 — RAG manual, sin framework: query.

Pregunta -> embedding -> retrieval en pgvector (SQL puro) -> prompt manual -> LLM.
"""

import os
import sys

import psycopg
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
TOP_K = 3

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
llm = Anthropic(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
)


def retrieve(question: str, k: int) -> list[tuple[str, float]]:
    query_embedding = voyage.embed([question], model="voyage-3.5", input_type="query").embeddings[0]

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

    return rows


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Respondé la pregunta usando SOLO la información del contexto.
Si el contexto no alcanza para responder, decilo explícitamente.

Contexto:
{context}

Pregunta: {question}"""


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "¿Qué experiencia tiene con Python?"

    rows = retrieve(question, TOP_K)
    print(f"--- chunks recuperados para: {question!r} ---")
    for content, distance in rows:
        print(f"[distancia={distance:.4f}] {content[:80]!r}...")

    prompt = build_prompt(question, [c for c, _ in rows])

    print("\n--- respuesta ---")
    with llm.messages.stream(
        model="deepseek-v4-flash",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
