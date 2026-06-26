"""
api.py — FastAPI service exposing MultiSent-RAG as a documented REST endpoint.

Run locally:
    uvicorn api:app --reload
Then open the interactive docs at  http://127.0.0.1:8000/docs

This reuses the SAME pipeline as the Gradio demo (core/reader.py): each request
checks the semantic cache first, and on a miss falls back to Chroma retrieval +
the Groq LLM. The response reports which path was taken, the latency, and the
evidence used — so the API is as transparent as the UI.
"""

from contextlib import asynccontextmanager
from time import perf_counter
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.reader import MultiSentRAG

# Pipeline is loaded ONCE at startup (not per request) and shared across calls.
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    print("Loading MultiSent-RAG pipeline...")
    _state["rag"] = MultiSentRAG()
    yield
    # --- shutdown ---
    _state.clear()


app = FastAPI(
    title="MultiSent-RAG API",
    description=(
        "Training-free **multilingual sentiment classification** across 12 languages, "
        "with a semantic **cache memory** and **cross-lingual** retrieval.\n\n"
        "Each request first checks the semantic cache; on a miss it retrieves similar "
        "labeled examples from a Chroma vector store and classifies with an LLM, then "
        "remembers the result. The response tells you which path was taken."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------- schemas ---------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="The sentence to classify, in any of the 12 supported languages.",
        examples=["J'adore ce produit, il fonctionne parfaitement !"],
    )
    language: Optional[str] = Field(
        None,
        description="Optional ISO-639-1 language hint (e.g. 'fr'). Inferred if omitted.",
        examples=["fr"],
    )


class RetrievedExample(BaseModel):
    text: str
    label: str
    language: str


class CacheHit(BaseModel):
    text: str
    label: str
    language: Optional[str] = None
    distance: float = Field(..., description="Angular distance to the matched entry (lower = closer).")


class PredictResponse(BaseModel):
    label: str = Field(..., description="positive | negative | neutral | uncertain")
    source: str = Field(..., description="'cache' if reused from memory, otherwise 'vector_db'.")
    latency_ms: float = Field(..., description="Server-side processing time, in milliseconds.")
    retrieved: Optional[List[RetrievedExample]] = Field(
        None, description="Examples retrieved (present on a fresh computation)."
    )
    cache_hit: Optional[CacheHit] = Field(
        None, description="The reused entry (present on a cache hit)."
    )


# --------------------------- routes ---------------------------
@app.get("/health", tags=["meta"])
def health():
    """Liveness probe — confirms the service is up and the model is loaded."""
    return {"status": "ok", "model_loaded": "rag" in _state}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """
    Classify the sentiment of a sentence.

    Returns the **label**, whether the answer came from the **semantic cache**
    or was **computed fresh**, the **latency**, and the **evidence** used.
    """
    rag = _state["rag"]
    t0 = perf_counter()
    result = rag.predict(req.text, language=req.language)
    latency_ms = (perf_counter() - t0) * 1000.0

    return PredictResponse(
        label=result["label"],
        source=result["source"],
        latency_ms=round(latency_ms, 1),
        retrieved=result.get("retrieved"),
        cache_hit=result.get("cache_hit"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)