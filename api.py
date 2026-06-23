"""
api.py — FastAPI backend for the Fraud Detection Engine

Endpoints:
    GET  /health          — sanity check
    POST /predict          — score a single transaction, return SHAP explanation
    GET  /stream           — SSE stream of scored transactions (live feed)
    GET  /stats            — running totals since server start

Run with:
    uvicorn api:app --reload --port 8000
"""

import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from simulator import transaction_stream

MODEL_DIR = "model"

# ── Globals populated at startup ──────────────────────────────────────────────

model: Optional[xgb.XGBClassifier] = None
explainer = None
feature_names: list = []
threshold: float = 0.5

stats = {
    "total": 0,
    "flagged": 0,
    "true_positives": 0,
    "false_positives": 0,
    "true_negatives": 0,
    "false_negatives": 0,
    "started_at": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer, feature_names, threshold

    model = xgb.XGBClassifier()
    model.load_model(f"{MODEL_DIR}/xgb_fraud.json")

    with open(f"{MODEL_DIR}/explainer.pkl", "rb") as f:
        explainer = pickle.load(f)

    with open(f"{MODEL_DIR}/feature_names.json") as f:
        feature_names = json.load(f)

    with open(f"{MODEL_DIR}/threshold.json") as f:
        threshold = json.load(f)["threshold"]

    stats["started_at"] = time.time()
    print(f"[api] Model loaded. Threshold={threshold:.2f}  Features={len(feature_names)}")

    yield  # app runs here

    print("[api] Shutting down.")


app = FastAPI(title="Fraud Detection Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project — fine to leave open
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class TransactionIn(BaseModel):
    features: Dict[str, float]


# ── Core scoring logic (shared by /predict and /stream) ──────────────────────

def score_transaction(features: dict) -> dict:
    """Run model inference + SHAP explanation for one transaction."""
    x = np.array([[features[name] for name in feature_names]])

    prob = float(model.predict_proba(x)[0, 1])
    is_fraud = prob >= threshold

    shap_values = explainer.shap_values(x)[0]
    contributions = sorted(
        zip(feature_names, shap_values.tolist()),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    top_features = [
        {"feature": name, "shap_value": round(val, 4), "input_value": round(features[name], 4)}
        for name, val in contributions[:5]
    ]

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": is_fraud,
        "top_features": top_features,
    }


def _update_stats(is_fraud_pred: bool, true_label: Optional[int]):
    stats["total"] += 1
    if is_fraud_pred:
        stats["flagged"] += 1

    if true_label is None:
        return  # no ground truth available (real /predict calls won't have one)

    if is_fraud_pred and true_label == 1:
        stats["true_positives"] += 1
    elif is_fraud_pred and true_label == 0:
        stats["false_positives"] += 1
    elif not is_fraud_pred and true_label == 0:
        stats["true_negatives"] += 1
    elif not is_fraud_pred and true_label == 1:
        stats["false_negatives"] += 1


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(tx: TransactionIn):
    missing = set(feature_names) - set(tx.features.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {sorted(missing)}")

    result = score_transaction(tx.features)
    _update_stats(result["is_fraud"], true_label=None)
    return result


@app.get("/stream")
def stream(fraud_rate: float = 0.015, delay: float = 1.2):
    """
    Server-Sent Events endpoint. Each event is a JSON-encoded transaction
    plus its model score and SHAP explanation.
    """

    def event_generator():
        for tx in transaction_stream(fraud_rate=fraud_rate, delay_seconds=delay):
            result = score_transaction(tx["features"])
            _update_stats(result["is_fraud"], true_label=tx["true_label"])

            payload = {
                "id": tx["id"],
                "merchant": tx["merchant"],
                "amount": tx["amount"],
                "true_label": tx["true_label"],
                **result,
            }
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stats")
def get_stats():
    tp, fp = stats["true_positives"], stats["false_positives"]
    tn, fn = stats["true_negatives"], stats["false_negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    uptime = time.time() - stats["started_at"] if stats["started_at"] else 0

    return {
        **stats,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "uptime_seconds": round(uptime, 1),
    }
