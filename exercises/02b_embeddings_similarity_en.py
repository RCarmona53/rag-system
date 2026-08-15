"""Fase 1.1 (variante) — mismo experimento que 02, pero en inglés.

Objetivo: ver si el "piso" de similaridad entre frases sin relación
cambia según el idioma (menos ruido gramatical compartido en inglés?).
"""

import os

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

SENTENCES = [
    "The dog runs in the park.",
    "A canine plays in the plaza.",
    "The stock market fell today.",
    "Equity markets had losses.",
    "I like coffee in the morning.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    result = client.embed(SENTENCES, model="voyage-3.5", input_type="document")
    vectors = [np.array(v) for v in result.embeddings]

    print(f"dimensión del embedding: {vectors[0].shape[0]}\n")

    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"[{sim:.3f}] {SENTENCES[i]!r} <-> {SENTENCES[j]!r}")


if __name__ == "__main__":
    main()
