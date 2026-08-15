# rag-system

Sistema de RAG (Retrieval-Augmented Generation) construido desde cero, sin frameworks primero y con LangChain después, como Proyecto 1 de un roadmap de portfolio para AI Engineer. Documento acá las decisiones técnicas y los bugs reales encontrados en el camino — no solo lo que funcionó, sino por qué.

Ver también [`GLOSARIO.md`](./GLOSARIO.md) — definiciones técnicas de todo lo visto en el proyecto, para repaso de entrevistas.

## Estado del roadmap

Este es el Proyecto 1 de 3 de un roadmap de portfolio (RAG → Agente con tools → AI SaaS completo).

- [x] Fase 1.1 — Fundamentos (LLM directo, embeddings, cosine similarity)
- [x] Fase 1.2 — RAG manual sin framework (chunking, pgvector, retrieval)
- [x] Fase 1.3 — API REST con FastAPI (errores, logging, costo)
- [x] Fase 1.4 — Migración a LangChain
- [x] Fase 1.5 — Evaluación con Ragas
- [ ] Proyecto 2 — Agente con tools (function calling, LangGraph, observabilidad)
- [ ] Proyecto 3 — AI SaaS completo (frontend, auth, streaming, Docker, CI)

## Qué hace

Subís un PDF (en este caso, un CV), el sistema lo procesa, genera embeddings, los guarda en `pgvector`, y respondés preguntas sobre el documento con respuestas basadas únicamente en el contenido recuperado (no en conocimiento general del modelo).

## Stack

- **LLM (chat):** DeepSeek `v4-flash`, vía el endpoint compatible con la API de Anthropic (`https://api.deepseek.com/anthropic`) — permite usar el SDK `anthropic` sin cambiar código.
- **Embeddings:** Voyage AI `voyage-3.5` (Anthropic no tiene API de embeddings propia).
- **Vector DB:** PostgreSQL + `pgvector`, en Docker.
- **API:** FastAPI.
- **Framework RAG:** pipeline manual primero (`exercises/03`-`04`), después reimplementado con LangChain (`exercises/05`) para comparar con criterio.
- **Evaluación:** Ragas, con el mismo DeepSeek como juez.

## Setup

```bash
docker compose up -d          # levanta Postgres + pgvector en localhost:5433
```

Crear `.env` en la raíz con:
```
DEEPSEEK_API_KEY=...
VOYAGE_API_KEY=...
```

```bash
uv run exercises/03_rag_ingest.py     # ingesta el PDF
uv run exercises/04_rag_query.py "tu pregunta"

# o la API completa:
uv run uvicorn app.main:app --port 8000
```

## Decisiones técnicas y lecciones (por fase)

### Fase 1.1 — Fundamentos

Antes de tocar el proyecto: llamadas directas al LLM sin framework (`exercises/01`) y embeddings + cosine similarity calculada a mano con `numpy` (`exercises/02`, `02b`).

**Hallazgo:** pares de frases *sin ninguna relación semántica* dieron cosine similarity de ~0.65-0.72, nunca cerca de 0 — probado en español e inglés, con el mismo resultado. Esto no es un bug, es una propiedad del modelo: oraciones cortas y declarativas comparten una región del espacio vectorial por su estructura, no por su significado.

**Por qué importa:** en RAG nunca se usa un umbral fijo de similaridad para decidir qué chunk es "relevante". Se usa **ranking relativo** (`ORDER BY ... LIMIT k`), porque el valor absoluto no es interpretable de esa forma.

### Fase 1.2 — RAG manual (sin framework)

**Chunking:** el primer intento fue por tamaño fijo de caracteres. Encontramos evidencia concreta de que corta palabras al medio (la palabra "used" quedó partida en "u" + "sed" entre dos chunks). Se resolvió agrupando por **oraciones completas** hasta un tamaño objetivo — con la particularidad de que el PDF de origen no tiene saltos de párrafo reales (pypdf pone casi cada palabra en su propia línea), así que "chunking por párrafo" clásico no era viable acá.

**Resultado medible:** después de arreglar el chunking, una pregunta sobre ".NET o Rails" pasó de recuperar una sola experiencia relevante a recuperar las dos correctas — chunks limpios → embeddings más representativos → mejor retrieval.

**Retrieval:** el operador nativo `<=>` de pgvector hace la búsqueda vectorial directo en SQL, sin traer todos los vectores a Python.

**Infraestructura:** Postgres + pgvector en Docker, con su propia red y volumen (`rag-system_default`, `rag-system_rag_pg_data`), aislado de otros proyectos con Postgres en la misma máquina.

### Fase 1.3 — API REST (FastAPI)

`POST /documents` (ingesta) y `POST /query` (pregunta → respuesta + fuentes + costo), con manejo de errores del proveedor (429/502 en vez de 500 crudo) y logging de costo/latencia por request. Precios verificados contra documentación oficial: DeepSeek `v4-flash` = $0.14/1M tokens input + $0.28/1M output; Voyage `voyage-3.5` = $0.06/1M tokens.

**Bug real:** `message.content[0].text` asumía que el primer bloque de la respuesta siempre era texto. DeepSeek devuelve primero un bloque de "thinking", así que `content[0]` a veces era ese bloque (sin atributo `.text`), rompiendo con un 500. Se arregló filtrando por `block.type == "text"` en vez de asumir la posición del bloque.

### Fase 1.4 — Migración a LangChain

Reimplementación del mismo pipeline con `langchain-anthropic`, `langchain-voyageai`, `langchain-postgres`, para comparar con criterio en vez de aceptar el framework como caja negra.

**Lo que resolvió solo:** manejo de la tabla en pgvector (colección propia, sin SQL manual), chunking sin cortar palabras (`RecursiveCharacterTextSplitter`), y — el hallazgo más interesante — el bug del bloque de "thinking" **no volvió a aparecer**, porque `ChatAnthropic` + `StrOutputParser` ya filtran esos bloques automáticamente.

**Lo que no resolvió:** la limpieza del texto crudo extraído del PDF (los chunks seguían con ruido de espacios), y el prompt de grounding, que se sigue escribiendo a mano igual.

**Bug que el framework introdujo:** la composición con `RunnableParallel` volvía a invocar el retriever sin que fuera obvio leyendo el código, duplicando una llamada a la API por pregunta — menos control sobre qué se ejecuta cuándo, el costo real de la abstracción.

**Nota de mantenimiento:** `langchain-community` (de donde viene `PyPDFLoader`) fue discontinuado en 2026. Se optó por seguir extrayendo el PDF con `pypdf` directo en vez de depender de un paquete sin mantenimiento.

### Fase 1.5 — Evaluación con Ragas

Dataset de 16 preguntas sobre el CV (incluyendo preguntas trampa — Python, Kubernetes, posgrados — para verificar que el sistema no alucine), evaluado con `Faithfulness`, `LLMContextRecall` y `FactualCorrectness`, usando el mismo DeepSeek como juez (sin depender de OpenAI).

**La corrida no salió limpia al primer intento:**
1. Con `max_tokens=500` en el juez, la mayoría de las evaluaciones fallaron por `LLMDidNotFinishException`. El primer promedio reportado ("1.0 perfecto") resultó ser `pandas.mean()` ignorando en silencio 13 de 16 filas fallidas — un promedio engañoso, no un resultado real.
2. Subir `max_tokens` a 4000 ayudó poco y aparecieron `TimeoutError` nuevos.
3. El fix real fue bajar la concurrencia del evaluador (`max_workers` de 16 a 3) — el modelo se saturaba con demasiadas llamadas en paralelo.

**Resultado final honesto:** `Faithfulness` y `Context Recall` se evaluaron con éxito en 11/16 y 15/16 preguntas, ambos con promedio 1.0 (cero alucinación detectada, incluso en las preguntas trampa). `FactualCorrectness` nunca se estabilizó (solo 3/16 completadas) — límite real de usar un modelo barato con thinking forzado como juez para una métrica que requiere descomponer la respuesta en varias afirmaciones verificables. Se documenta como limitación conocida en vez de forzar un número.

**El hallazgo de mayor valor:** una fila con `factual_correctness=0.00` no era un problema de la métrica — era una **respuesta vacía real** del sistema en producción. Mismo bug del bloque de "thinking" de la Fase 1.3, pero esta vez agotando los 300 tokens de presupuesto de generación sin dejar espacio para el texto de la respuesta. Se corrigió subiendo `max_tokens` a 1024 en la generación y agregando una validación explícita que ahora devuelve un 502 claro en vez de una respuesta vacía silenciosa.

## Limitaciones conocidas

- `FactualCorrectness` de Ragas no es confiable con DeepSeek `v4-flash` como juez vía este endpoint compatible con Anthropic — métrica documentada pero no usada para decisiones de calidad.
- El PDF de prueba es de una sola página; el pipeline no fue probado con documentos grandes ni con chunking a gran escala.
- Sin autenticación ni rate limiting propio en la API (planeado para el Proyecto 3).
