"""Fase 1.4 — el mismo RAG de exercises/03 y 04, reimplementado con LangChain.

Comparar contra el pipeline manual: ¿qué resolvió el framework solo,
y qué seguís teniendo que decidir vos igual?

Nota: langchain-community (donde vivía PyPDFLoader) fue discontinuado en 2026,
así que seguimos extrayendo el PDF con pypdf directo, como en exercises/03,
y usamos LangChain solo para lo que sí sigue mantenido: splitter, vectorstore y chain.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pypdf import PdfReader

load_dotenv()

PDF_PATH = "/home/rodrigo/Descargas/Resume-Rodrigo-Carmona.pdf"
CONNECTION = "postgresql+psycopg://rag:rag@localhost:5433/rag"
COLLECTION_NAME = "cv_langchain"

embeddings = VoyageAIEmbeddings(model="voyage-3.5")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION,
    use_jsonb=True,
)

llm = ChatAnthropic(
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/anthropic",
    max_tokens=300,
)

PROMPT = ChatPromptTemplate.from_template("""Respondé la pregunta usando SOLO la información del contexto.
Si el contexto no alcanza para responder, decilo explícitamente.

Contexto:
{context}

Pregunta: {question}""")


def load_pdf(path: str) -> list[Document]:
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() for page in reader.pages)
    return [Document(page_content=text, metadata={"source": path})]


def ingest() -> None:
    docs = load_pdf(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)
    print(f"{len(chunks)} chunks insertados (LangChain) en la colección {COLLECTION_NAME!r}")


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def query(question: str) -> None:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(question)
    print(f"--- chunks recuperados para: {question!r} ---")
    for d in docs:
        print(repr(d.page_content[:80]))

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    print("\n--- respuesta ---")
    print(answer)


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        ingest()
    else:
        question = sys.argv[1] if len(sys.argv) > 1 else "¿Qué experiencia tiene con Rails?"
        query(question)
