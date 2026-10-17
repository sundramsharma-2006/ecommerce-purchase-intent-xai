"""FastAPI service: `uvicorn src.api.app:app --port 8000` (docs at /docs).

The model + explainer are loaded once at startup (lifespan), never per request.
NOTE: written to the FastAPI/Pydantic v2 API; it could not be started in the build sandbox (fastapi not
installable there). The scoring logic it calls (src/serving/predictor.py) is unit-tested.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import HealthOut, PredictionOut, SessionIn
from src.serving.predictor import ModelNotFoundError, PurchasePredictor
from src.utils.logging_utils import get_logger

log = get_logger("api")
STATE: dict = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["predictor"] = PurchasePredictor.load()
    except ModelNotFoundError as exc:
        log.error("%s", exc)          # the service still starts so /health can report model_loaded=false
    yield
    STATE["predictor"] = None


app = FastAPI(title="Purchase-Intent API", version="1.0.0", lifespan=lifespan,
              description="Scores a shopping session for purchase likelihood and explains the score.")


@app.middleware("http")
async def structured_logging(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info(json.dumps({"path": request.url.path, "method": request.method, "status": response.status_code,
                         "ms": round((time.perf_counter() - t0) * 1000, 1)}))
    return response


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    p = STATE["predictor"]
    return HealthOut(status="ok", model_loaded=p is not None, model=p.meta.get("model") if p else None)


@app.post("/predict", response_model=PredictionOut)
def predict(session: SessionIn, explain: bool = True) -> PredictionOut:
    p: PurchasePredictor | None = STATE["predictor"]
    if p is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first: python -m src.train")
    try:
        return PredictionOut(**p.predict(session.model_dump(), explain=explain))
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/model-info")
def model_info() -> dict:
    p = STATE["predictor"]
    if p is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {**p.meta, "threshold": p.threshold}
