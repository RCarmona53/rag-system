"""Fase 1.1 — llamada directa a la API de Anthropic, sin framework.

Objetivo: entender el ciclo system/user -> streaming -> temperature
antes de que cualquier framework lo esconda.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
)

SYSTEM_PROMPT = "Sos un asistente conciso. Respondé en una sola oración."


def ask(question: str, temperature: float = 1.0) -> None:
    print(f"\n--- temperature={temperature} ---")
    with client.messages.stream(
        model="deepseek-v4-flash",
        max_tokens=200,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

    final = stream.get_final_message()
    print(f"tokens: input={final.usage.input_tokens} output={final.usage.output_tokens}")


if __name__ == "__main__":
    question = "¿Por qué el cielo es azul?"
    ask(question, temperature=0.0)
    ask(question, temperature=1.0)
