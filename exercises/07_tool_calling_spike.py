"""Fase 0 (Proyecto 2) — spike: ¿tool-calling de LangChain funciona con DeepSeek
en el endpoint compatible con Anthropic, antes de construir el agente completo?

Gatea todo el resto del build (`app/agent.py`). Si algo de esto falla, NO se sigue
con el diseño original sin documentar qué pasó — ver README.md, sección Proyecto 2,
"Spike de tool-calling", para el resultado real (haya salido bien o mal) y, si
falló, qué opción de la escalera de fallback (design ADR, sección "Fallback ladder")
se terminó usando.

Corre standalone, sin pytest: `uv run exercises/07_tool_calling_spike.py`.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética simple, ej '6*7'."""
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — spike descartable, no es el evaluador final


llm = ChatAnthropic(
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
    max_tokens=300,
)
llm_with_tools = llm.bind_tools([calculator])


def assertion_1_tool_call_parsed() -> None:
    """El modelo debe emitir un tool_call con argumentos parseados para una pregunta de matemática."""
    response = llm_with_tools.invoke([HumanMessage(content="¿Cuánto es 12*7? Usá la herramienta calculator.")])
    assert response.tool_calls, f"esperaba tool_calls, la respuesta vino vacía de tool_calls: {response}"
    call = response.tool_calls[0]
    assert call["name"] == "calculator", f"esperaba 'calculator', vino {call['name']!r}"
    assert "expression" in call["args"], f"esperaba arg 'expression', vinieron {call['args'].keys()}"
    print(f"[1/3] OK — tool_call parseado: {call}")


def assertion_2_roundtrip_final_text() -> None:
    """Alimentar el ToolMessage de vuelta debe producir una respuesta de texto final."""
    human = HumanMessage(content="¿Cuánto es 12*7? Usá la herramienta calculator.")
    ai = llm_with_tools.invoke([human])
    assert ai.tool_calls, "assertion 2 depende de que 1 haya funcionado (sin tool_calls no hay roundtrip)"
    call = ai.tool_calls[0]
    tool_result = calculator.invoke(call["args"])
    tool_msg = ToolMessage(content=tool_result, tool_call_id=call["id"])
    final = llm_with_tools.invoke([human, ai, tool_msg])
    assert final.content, f"esperaba texto final no vacío tras el roundtrip, vino: {final}"
    print(f"[2/3] OK — respuesta final tras roundtrip: {final.content!r}")


def assertion_3_no_forced_call() -> None:
    """Una pregunta que no necesita ninguna tool no debe forzar un tool_call."""
    response = llm_with_tools.invoke([HumanMessage(content="¿De qué color es el cielo en un día despejado?")])
    assert not response.tool_calls, f"esperaba 0 tool_calls, el modelo llamó una tool igual: {response.tool_calls}"
    assert response.content, "esperaba texto en la respuesta sin tool"
    print(f"[3/3] OK — sin tool_calls, respuesta directa: {response.content!r}")


if __name__ == "__main__":
    try:
        assertion_1_tool_call_parsed()
        assertion_2_roundtrip_final_text()
        assertion_3_no_forced_call()
    except Exception as e:
        print(f"\nSPIKE FALLÓ: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    print("\nSPIKE OK — bind_tools() funciona con DeepSeek v4-flash en el endpoint compatible con Anthropic.")
