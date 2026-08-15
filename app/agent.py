"""Proyecto 2 — agente con tool-calling (LangGraph) sobre el RAG del Proyecto 1.

Grafo explícito (no `create_react_agent`, ver README ADR D1): nodo `agent`
(LLM con tools bindeadas) -> `ToolNode` -> edge condicional que decide si
seguir llamando tools, terminar, o cortar por límite de pasos.

`app/rag.py` se importa pero nunca se modifica — `rag_search` es un adapter
fino sobre `answer()`, no una reimplementación.
"""

import ast
import json
import logging
import operator
import os
import time
import uuid
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.rag import DATABASE_URL, DEEPSEEK_INPUT_PRICE_PER_M, DEEPSEEK_OUTPUT_PRICE_PER_M
from app.rag import answer as rag_answer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

MAX_STEPS = 4  # ADR D4 — cap de dominio, capa 1 (observable/testeable)
RECURSION_LIMIT = 10  # ADR D4 — backstop de LangGraph, capa 2

# ADR D5 — gate de LangSmith en el import: si no hay API key, no-op real
# (nunca se construye un tracer, nunca sale una llamada de red).
if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "rag-system-agent")
    TRACING_ENABLED = True
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    TRACING_ENABLED = False


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


# ---------------------------------------------------------------------------
# Tool 1: calculadora — ADR D6, evaluador AST whitelist, jamás eval()/exec()
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPONENT = 1000  # acota `**` para evitar cómputo/memoria desmedidos


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponente fuera de rango: {right}")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"nodo no permitido: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    """Evalúa una expresión aritmética usando solo un whitelist de nodos AST.

    Permite únicamente literales numéricos y los operadores de arriba.
    Sin nombres, sin atributos, sin llamadas, sin builtins — nunca `eval()`
    ni `exec()`. Cualquier otro nodo levanta `ValueError`.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"sintaxis inválida: {e}") from e
    return _eval_node(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión aritmética simple, ej '6*7' o '(3+4)/2'.

    Soporta + - * / // % ** y paréntesis. No admite variables, funciones
    ni texto: solo números y operadores.
    """
    try:
        result = _safe_eval(expression)
    except (ValueError, ZeroDivisionError, TypeError, RecursionError) as e:
        return f"error: expresión inválida ({e})"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# Tool 2: rag_search — adapter fino sobre app.rag.answer() (ADR D3)
# ---------------------------------------------------------------------------


@tool
def rag_search(question: str) -> str:
    """Busca en los documentos ingeridos (PDF cargado vía POST /documents)
    y responde la pregunta usando RAG. Usar solo para preguntas sobre el
    contenido de esos documentos, no para matemática ni charla general.
    """
    result = answer(question)
    return json.dumps({"answer": result["answer"], "cost_usd": result["cost_usd"]})


# alias module-level para que rag_search siga la llamada indirecta —
# permite monkeypatchear `app.agent.answer` en los tests sin pegarle
# a Voyage/Postgres/DeepSeek (ver tests/test_agent_tools.py).
answer = rag_answer

TOOLS = [rag_search, calculator]

# ---------------------------------------------------------------------------
# LLM + grafo (ADR D1 — StateGraph hecho a mano, no create_react_agent)
# ---------------------------------------------------------------------------

llm = ChatAnthropic(
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
    max_tokens=1024,
)
llm_with_tools = llm.bind_tools(TOOLS)

# Bug real encontrado en pruebas manuales: sin este system prompt, una
# pregunta como "¿qué experiencia tiene con Rails?" el modelo la interpretaba
# como "¿qué experiencia tenés VOS (el asistente)?" y respondía de memoria en
# vez de llamar a rag_search — violaba el requisito de ruteo del spec. Se
# arma acá (no se guarda en el checkpoint) para no duplicarse turno a turno.
SYSTEM_PROMPT = SystemMessage(
    content=(
        "Sos un asistente que responde preguntas sobre una persona a partir de "
        "su CV, que está cargado como documento, y que también puede hacer "
        "cálculos matemáticos simples. Tenés dos herramientas: `rag_search`, que "
        "busca en el CV cargado — usala SIEMPRE que te pregunten por la "
        "experiencia, habilidades, formación, trayectoria o cualquier dato "
        "biográfico o profesional de esa persona, en vez de responder de "
        "memoria o de forma genérica; y `calculator`, para cualquier operación "
        "aritmética. Si la pregunta no necesita ninguna de las dos, respondé "
        "directamente."
    )
)


def _agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT, *messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    if state.get("steps", 0) >= MAX_STEPS:
        return "cap"
    return "tools"


def build_graph(checkpointer: Any = None):
    """Construye y compila el StateGraph. `checkpointer=None` -> grafo sin
    persistencia (útil para tests unitarios que no tocan `run_agent`)."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    # handle_tool_errors=False: las excepciones de la tool (ValueError,
    # LookupError, etc. de app.rag.answer) deben propagar hasta la ruta
    # /agent para el mapeo de errores, no quedar absorbidas como un
    # ToolMessage de error silencioso.
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=False))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END, "cap": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Checkpointer (ADR D2) — PostgresSaver sobre la misma DB de pgvector,
# fallback a InMemorySaver si Postgres no está disponible al arrancar.
# ---------------------------------------------------------------------------

_checkpointer_cm = None
_graph = None


def setup_checkpointer():
    """Llamado desde el lifespan de FastAPI. Devuelve el grafo compilado."""
    global _checkpointer_cm, _graph
    try:
        _checkpointer_cm = PostgresSaver.from_conn_string(DATABASE_URL)
        checkpointer = _checkpointer_cm.__enter__()
        checkpointer.setup()
    except Exception as e:
        logger.warning(
            "no se pudo conectar PostgresSaver (%s); usando InMemorySaver "
            "(solo válido con un único worker de uvicorn)",
            e,
        )
        _checkpointer_cm = None
        checkpointer = InMemorySaver()
    _graph = build_graph(checkpointer)
    return _graph


def teardown_checkpointer():
    global _checkpointer_cm, _graph
    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)
    _checkpointer_cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("el grafo no fue inicializado — ¿corrió el lifespan de FastAPI?")
    return _graph


# ---------------------------------------------------------------------------
# Extracción de texto y costo (ADR D7) — mismo bug de "content blocks" que
# app/rag.py: el bloque final puede venir como lista tipada (thinking+text),
# confirmado en exercises/07_tool_calling_spike.py.
# ---------------------------------------------------------------------------


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def _sum_llm_cost(messages: list[AnyMessage]) -> float:
    cost = 0.0
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            input_tokens = m.usage_metadata.get("input_tokens", 0) or 0
            output_tokens = m.usage_metadata.get("output_tokens", 0) or 0
            cost += (input_tokens / 1_000_000) * DEEPSEEK_INPUT_PRICE_PER_M
            cost += (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PRICE_PER_M
    return cost


def _sum_tool_cost(messages: list[AnyMessage]) -> float:
    cost = 0.0
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            parsed = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "cost_usd" in parsed:
            cost += float(parsed["cost_usd"])
    return cost


def run_agent(question: str, thread_id: str | None = None) -> dict:
    """Corre el grafo para una pregunta y devuelve la respuesta con
    tools usadas, costo y latencia del turno actual (no del thread entero)."""
    thread_id = thread_id or str(uuid.uuid4())
    graph = get_graph()
    start = time.monotonic()

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    result = graph.invoke({"messages": [HumanMessage(content=question)], "steps": 0}, config=config)

    messages: list[AnyMessage] = result["messages"]
    turn_start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn_messages = messages[turn_start:]

    tools_used: list[str] = []
    for m in turn_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tools_used.extend(tc["name"] for tc in m.tool_calls)

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # se cortó por MAX_STEPS con un tool_call pendiente: no hay texto final real
        answer_text = "Se alcanzó el límite de pasos del agente sin completar una respuesta final."
    else:
        answer_text = _extract_text(last.content)

    cost_usd = round(_sum_llm_cost(turn_messages) + _sum_tool_cost(turn_messages), 6)
    latency_ms = (time.monotonic() - start) * 1000

    return {
        "answer": answer_text,
        "tools_used": tools_used,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "thread_id": thread_id,
    }


__all__ = [
    "AgentState",
    "GraphRecursionError",
    "TRACING_ENABLED",
    "answer",
    "build_graph",
    "calculator",
    "rag_search",
    "run_agent",
    "setup_checkpointer",
    "teardown_checkpointer",
]
