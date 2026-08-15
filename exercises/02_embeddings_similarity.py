"""Fase 1.1 — embeddings y cosine similarity a mano, sin librerías de vectores.

Objetivo: ver con los propios ojos que un embedding es solo un vector,
y que "similaridad semántica" es una cuenta de trigonometría.
"""

import os

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

SENTENCES = [
    "El perro corre en el parque.",
    "Un can juega en la plaza.",
    "La bolsa de valores cayó hoy.",
    "El mercado accionario tuvo pérdidas.",
    "Me gusta el café por la mañana.",
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
