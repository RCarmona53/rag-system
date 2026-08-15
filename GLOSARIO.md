# Glosario técnico — AI Engineering

Definiciones de los conceptos vistos en este proyecto, para repasar antes de una entrevista. Cada término incluye, cuando aplica, el ejemplo concreto donde lo vimos funcionar (o romperse) en este mismo repo.

## Fundamentos de LLMs

**Token** — la unidad mínima de texto que procesa un LLM. No es una palabra ni un carácter: puede ser una palabra completa, un pedazo de palabra, o un signo de puntuación, según cómo el modelo particionó su vocabulario. El costo de las APIs se cobra por token, no por palabra ni por carácter.

**Context window** — la cantidad máxima de tokens (entrada + salida combinados) que un modelo puede "ver" en una sola llamada. Todo lo que no entra ahí, el modelo directamente no lo sabe, aunque esté en el prompt.

**Temperature** — controla cuánto azar se permite al elegir el siguiente token. `temperature=0` → salida casi determinística (la vimos repetirse casi igual en cada corrida). `temperature=1` → más variación entre corridas. No cambia si el modelo "sabe más", cambia cuánto explora entre respuestas igual de probables.

**System prompt / user message** — el system prompt define el comportamiento general del modelo para toda la conversación (tono, reglas, rol); el user message es la pregunta o instrucción puntual. Se envían como mensajes separados, no concatenados en un solo string.

**Streaming** — recibir la respuesta del modelo token por token (o en pedacitos) a medida que se genera, en vez de esperar la respuesta completa. Mejora la latencia percibida por el usuario; lo usamos en `01_chat_direct.py`.

**Content blocks / thinking block** — la respuesta de un LLM moderno no es un string plano: es una lista de bloques tipados (`text`, `thinking`, `tool_use`, etc.). Algunos modelos (como DeepSeek con reasoning activado) devuelven un bloque de "pensamiento" **antes** del bloque de texto final. Asumir que el primer bloque siempre es texto (`message.content[0].text`) fue el bug real que nos rompió la API dos veces en este proyecto — hay que filtrar explícitamente por `block.type == "text"`.

**max_tokens** — el límite de tokens que el modelo puede generar en la respuesta. Si el modelo gasta ese presupuesto "pensando" (bloque `thinking`), puede quedarse sin espacio para el texto final y devolver una respuesta vacía — nos pasó literalmente en la Fase 1.5.

## Embeddings y similaridad

**Embedding** — un vector (lista de números, en este proyecto de 1024 dimensiones con `voyage-3.5`) que representa el significado semántico de un texto. Textos con significado parecido generan vectores que apuntan en direcciones parecidas.

**Cosine similarity** — mide el ángulo entre dos vectores (no su magnitud): `(A·B) / (‖A‖·‖B‖)`. Va de -1 a 1; cerca de 1 significa "misma dirección semántica". En este proyecto confirmamos que pares de frases sin relación dan ~0.65-0.72, no cerca de 0 — el "piso" depende de la estructura del texto, no solo del significado.

**Cosine distance** — `1 - cosine_similarity`. Es lo que usa el operador `<=>` de pgvector: menor distancia = más similar. Por eso el retrieval ordena `ORDER BY embedding <=> query ASC`.

**Dot product / norma (norm)** — el producto punto suma el producto de cada componente de dos vectores; la norma es su "largo" (magnitud). Dividir el dot product por las normas es lo que convierte una medida cruda en algo comparable entre vectores de distinto tamaño — es literalmente la fórmula de cosine similarity.

## RAG (Retrieval-Augmented Generation)

**RAG** — arquitectura donde, antes de responder, el sistema recupera información relevante de una fuente externa (documentos propios) y se la pasa al LLM como contexto, en vez de confiar solo en lo que el modelo aprendió en su entrenamiento. Reduce alucinaciones y permite responder sobre datos privados o actualizados.

**Chunking** — dividir un documento largo en pedazos más chicos antes de generar embeddings, porque un embedding de un documento entero mezclaría demasiados temas distintos en un solo vector. El *cómo* se corta importa: por tamaño fijo de caracteres puede cortar palabras al medio (lo vimos pasar); por oraciones completas respeta unidades de sentido.

**Overlap** — solapar un poco de contenido entre chunks consecutivos, para no perder contexto en una idea que quedó justo en el borde de un corte.

**Grounding** — la práctica de instruir al modelo a responder *solo* con la información del contexto recuperado ("si no está en el contexto, decilo"), en vez de rellenar con conocimiento general. Es lo que separa un RAG bien armado de un chatbot que alucina con confianza.

**Retrieval** — el paso de buscar y traer los chunks más relevantes para una pregunta, típicamente por similaridad vectorial.

**Vector database** — una base de datos optimizada para guardar vectores y buscar por similaridad de forma eficiente. Usamos PostgreSQL con la extensión `pgvector`, que agrega el tipo `VECTOR` y operadores de distancia (`<=>`, `<->`) directo en SQL.

**HNSW (Hierarchical Navigable Small World)** — el tipo de índice que usa `pgvector` para hacer la búsqueda de vecinos más cercanos rápido en vectores de alta dimensión, en vez de comparar contra todos los vectores uno por uno.

**top-k** — cuántos resultados (chunks) trae el retrieval para cada pregunta. Es un parámetro de diseño: muy bajo pierde contexto, muy alto diluye el prompt con información irrelevante.

## Frameworks

**LangChain** — librería que da abstracciones estándar para las piezas de una app de LLM: cargar documentos, chunking, embeddings, vector stores, retrievers, y cadenas que las conectan. Ahorra código repetitivo, pero también esconde detalles — vale la pena saber hacer el pipeline a mano primero para poder evaluar qué te está resolviendo de verdad.

**LCEL (LangChain Expression Language)** — la sintaxis de LangChain para encadenar componentes con el operador `|` (ej: `prompt | llm | parser`). Cada paso es un `Runnable` que se puede invocar, componer o correr en paralelo.

**Document loader** — componente que carga un archivo (PDF, web, etc.) y lo convierte en objetos `Document` que el resto del pipeline puede procesar.

**Text splitter** — el componente de chunking de LangChain (ej: `RecursiveCharacterTextSplitter`), que intenta cortar por separadores naturales (párrafos, líneas, espacios) antes de recurrir a un corte crudo por caracteres.

**VectorStore** — la abstracción de LangChain sobre una base de datos vectorial (en este caso, `PGVector` sobre nuestro mismo Postgres). Maneja la creación de tablas y la inserción de embeddings sin que escribas SQL.

**Retriever** — un `Runnable` de LangChain que envuelve al VectorStore y expone `.invoke(pregunta)` para traer los chunks más relevantes — la versión encapsulada del `SELECT ... ORDER BY ... LIMIT k` que escribimos a mano.

## Producción / backend de IA

**Endpoint** — una URL específica de una API que responde a un tipo de request (ej: `POST /query`).

**Rate limit** — el límite de cuántas requests por minuto/segundo permite un proveedor. Lo sufrimos en carne propia con el free tier de Voyage (3 RPM) — la solución no fue "esperar más", fue **batchear** las llamadas en una sola request en vez de una por pregunta.

**Timeout** — el tiempo máximo que se espera una respuesta antes de abortar la request. Puede fallar por el timeout del cliente HTTP o por un `RunConfig` propio de una herramienta (como Ragas) — son cosas distintas y hay que saber cuál está fallando.

**Latencia** — el tiempo que tarda una request en completarse. La medimos en cada endpoint (`latency_ms`) porque en producción importa tanto como la corrección de la respuesta.

**Logging / observabilidad** — registrar qué pasó en cada request (pregunta, chunks usados, tokens, costo, latencia) para poder debuggear después sin tener que reproducir el problema en vivo.

**Costo por token** — las APIs de LLM y embeddings cobran por millón de tokens, con precios distintos para input y output (y a veces caché). Calcular el costo real de cada request (`tokens × precio`) es lo que permite responder "¿cuánto me sale este sistema por mes?" en una entrevista, no solo "funciona".

## Evaluación

**Evaluation dataset** — un conjunto de preguntas con respuesta esperada (`reference`), usado para medir la calidad del sistema de forma repetible, en vez de probar preguntas sueltas a ojo.

**LLM-as-judge** — usar un LLM para evaluar la calidad de la respuesta de *otro* LLM (o del mismo sistema), en vez de un humano. Barato y escalable, pero no infalible — en este proyecto encontramos que la confiabilidad del juez depende mucho del modelo elegido y de cómo maneja "thinking" bajo prompts que exigen salida estructurada.

**Faithfulness** — métrica que mide si la respuesta contiene *solo* afirmaciones respaldadas por el contexto recuperado. Es la métrica que detecta alucinación directamente.

**Context Recall** — métrica que mide si el retrieval trajo toda la información necesaria para responder correctamente, comparado contra la respuesta de referencia. Mide la calidad del *retrieval*, no de la generación.

**Factual Correctness** — métrica que compara la respuesta generada contra la referencia y mide qué tan correcta es factualmente (típicamente descomponiendo la respuesta en afirmaciones individuales y verificando cada una). Es la métrica más costosa de calcular — y la que menos se sostuvo con un juez barato en este proyecto.

**Hallucination (alucinación)** — cuando un modelo genera información que suena plausible pero no está respaldada por el contexto ni es verdadera. El objetivo central de un buen grounding y de medir `Faithfulness` es detectar y minimizar esto.

**Ground truth / reference** — la respuesta "correcta" contra la que se compara la salida del sistema en una evaluación. Sin esto, no hay forma objetiva de medir mejora entre versiones.

**Regression testing (de calidad)** — volver a correr el dataset de evaluación después de un cambio (nuevo prompt, nuevo modelo, nuevo chunking) para confirmar que la calidad no empeoró — el equivalente de un test suite, pero para comportamiento de IA en vez de lógica determinística.

**Concurrency / max_workers** — cuántas llamadas en paralelo se hacen durante una evaluación (o cualquier proceso batch). Más concurrencia es más rápido, pero puede saturar al proveedor y generar timeouts que no tienen nada que ver con el contenido de las llamadas — lo vivimos al bajar `max_workers` de 16 a 3 y ver desaparecer los `TimeoutError`.
