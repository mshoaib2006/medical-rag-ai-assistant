from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core import MedicalRAG

app = FastAPI(title="Medical RAG API", version="2.0")

#Local dev CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MedicalRAG()


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(query: Query):
    result = rag.answer(query.question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "normalized_query": result.normalized_query,
    }


#uvicorn api:app --reload