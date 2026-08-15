"""Fase 1.5 — evaluación básica del RAG con Ragas.

Dataset chico (preguntas + respuesta esperada) corrido contra el pipeline
manual (app/rag.py) y evaluado con 3 métricas usando DeepSeek como juez
(no OpenAI): Faithfulness, LLMContextRecall, FactualCorrectness.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from app.rag import DATABASE_URL, build_prompt, llm, voyage

load_dotenv()

evaluator_llm = LangchainLLMWrapper(
    ChatAnthropic(
        model="deepseek-v4-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/anthropic",
        max_tokens=4000,
        timeout=120,
    )
)

# max_workers bajo a propósito: con 16 en paralelo (el default), un modelo
# "flash" con thinking activado se satura y empieza a tirar TimeoutError
# que no tiene nada que ver con max_tokens.
RUN_CONFIG = RunConfig(max_workers=3, timeout=150)

DATASET = [
    ("¿En qué empresa trabaja actualmente Rodrigo?",
     "Trabaja en Concentrix como Fullstack Ruby on Rails Developer desde 2022."),
    ("¿Qué hace en su rol actual?",
     "Desarrollo backend con Rails y REST APIs para una plataforma de salud, y un servicio en la nube con AWS Step Functions, Lambda y Amazon Bedrock para automatizar extracción de datos de PDFs."),
    ("¿Dónde trabajó antes de Concentrix?",
     "En Maer Software como .Net Backend Developer, de 2021 a 2022."),
    ("¿Y antes de Maer Software?",
     "En Drogeria Saporitti como .Net Backend Developer, de 2020 a 2021."),
    ("¿Qué tecnología usa para testing?",
     "RSpec."),
    ("¿Qué usa para procesamiento asíncrono?",
     "Sidekiq y Redis."),
    ("¿Qué certificación tiene?",
     "AWS Certified AI Practitioner, obtenida en 2025."),
    ("¿Dónde estudió?",
     "En el Instituto Terciario de Tecnología (ISTEA), como Técnico en Desarrollo de Software, de 2020 a 2023."),
    ("¿Qué bases de datos maneja?",
     "PostgreSQL, SQL Server y Redis."),
    ("¿Qué nivel de inglés tiene?",
     "Avanzado."),
    ("¿Con qué empresa relacionada a Heineken trabajó?",
     "The Next Ad, una empresa basada en los Países Bajos, durante su paso por Drogeria Saporitti."),
    ("¿Qué framework de frontend usó en Drogeria Saporitti?",
     "React con Redux."),
    ("¿Qué experiencia tiene con Python?",
     "El CV no menciona experiencia con Python."),
    ("¿Tiene experiencia con Kubernetes?",
     "El CV no menciona experiencia con Kubernetes."),
    ("¿Tiene un doctorado o una maestría?",
     "El CV no menciona ningún posgrado, solo el título de Técnico en Desarrollo de Software de ISTEA."),
    ("¿Qué servicios de AWS usa?",
     "S3, Lambda, Step Functions, Bedrock, EventBridge, API Gateway, CloudWatch, Textract, IAM y DynamoDB."),
]


def build_dataset() -> EvaluationDataset:
    questions = [q for q, _ in DATASET]
    embed_result = voyage.embed(questions, model="voyage-3.5", input_type="query")

    rows = []
    with psycopg.connect(DATABASE_URL) as conn:
        for (question, reference), embedding in zip(DATASET, embed_result.embeddings):
            db_rows = conn.execute(
                """
                SELECT content, embedding <=> %s AS distance
                FROM chunks
                ORDER BY distance
                LIMIT 3
                """,
                (str(embedding),),
            ).fetchall()
            chunks = [content for content, _ in db_rows]

            prompt = build_prompt(question, chunks)
            message = llm.messages.create(
                model="deepseek-v4-flash",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = "".join(b.text for b in message.content if b.type == "text")

            rows.append({
                "user_input": question,
                "retrieved_contexts": chunks,
                "response": response_text,
                "reference": reference,
            })
            print(f"[ok] {question}")

    return EvaluationDataset.from_list(rows)


def main() -> None:
    dataset = build_dataset()

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), LLMContextRecall(), FactualCorrectness()],
        llm=evaluator_llm,
        run_config=RUN_CONFIG,
    )

    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)

    print("\n--- promedios ---")
    print(df[["faithfulness", "context_recall", "factual_correctness(mode=f1)"]].mean())

    print("\n--- peores 3 por faithfulness (posible alucinación) ---")
    print(df.sort_values("faithfulness").head(3)[["user_input", "faithfulness", "response"]])


if __name__ == "__main__":
    main()
