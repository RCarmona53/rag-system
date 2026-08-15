"""Smoke tests para las dos tools del agente (Proyecto 2).

Alcance minimo acordado: sin red, sin Postgres, sin llamadas al LLM.
Cubre especificamente el caso de amenaza del ADR D6 (calculadora sin eval()).
"""

import json

import pytest

from app.agent import calculator, rag_search


def test_calculator_valid_expression():
    assert calculator.invoke({"expression": "6*7"}) == "42"


def test_calculator_hostile_input_is_rejected():
    """ADR D6 — el evaluador es un whitelist de AST, nunca eval()/exec().

    Una expresion hostil que intentaria ejecutar codigo arbitrario si el
    evaluador usara eval() debe ser rechazada (excepcion o string de error),
    y en ningun caso debe ejecutarse.
    """
    hostile = "__import__('os').system('echo pwned')"
    result = calculator.invoke({"expression": hostile})
    assert "pwned" not in result
    assert "error" in result.lower() or "inválid" in result.lower() or "invalid" in result.lower()


def test_calculator_rejects_names_and_attributes():
    result = calculator.invoke({"expression": "os.system('ls')"})
    assert "error" in result.lower() or "inválid" in result.lower() or "invalid" in result.lower()


def test_rag_search_returns_parseable_json(monkeypatch):
    """rag_search envuelve app.rag.answer(); acá se stubea para no pegarle
    a Voyage/Postgres/DeepSeek."""

    def fake_answer(question: str, k: int = 3) -> dict:
        return {
            "answer": "respuesta de prueba",
            "sources": [{"content": "chunk de prueba", "distance": 0.1}],
            "usage": {"embedding_tokens": 10, "input_tokens": 20, "output_tokens": 5},
            "cost_usd": 0.000123,
        }

    monkeypatch.setattr("app.agent.answer", fake_answer)

    raw = rag_search.invoke({"question": "¿qué experiencia tiene con Rails?"})
    parsed = json.loads(raw)
    assert parsed["answer"] == "respuesta de prueba"
    assert parsed["cost_usd"] == pytest.approx(0.000123)
