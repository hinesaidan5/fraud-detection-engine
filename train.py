"""
train.py — Train XGBoost fraud detection model + persist SHAP explainer

Usage:
    python train.py --data ../data/creditcard.csv

Dataset: Kaggle "Credit Card Fraud Detection"
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Outputs (written to ./model/):
    xgb_fraud.json        — trained XGBoost model
    explainer.pkl         — SHAP TreeExplainer (with background sample baked in)
    feature_names.json    — ordered feature list used at inference time
    threshold.json         — F1-tuned classification threshold
"""

import argparse
import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[train] Loading dataset from {path}")
    df = pd.read_csv(path)
    print(f"[train] Shape: {df.shape}  |  Fraud rate: {df['Class'].mean():.4%}")
    return df


def preprocess(df: pd.DataFrame):
    """Scale Amount/Time; drop raw columns; return X, y, feature_names."""
    df = df.copy()

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values
    y = df["Class"].values
    return X, y, feature_cols


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val):
    """Train XGBoost with scale_pos_weight to handle class imbalance (~0.17% fraud)."""
    neg, pos = np.bincount(y_train)
    spw = neg / pos
    print(f"[train] scale_pos_weight = {spw:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def tune_threshold(model, X_val, y_val):
    """Find the probability threshold that maximizes F1 on the validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"[train] Best threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return float(best_t)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    print(f"\n[eval] ROC-AUC: {roc:.4f}  |  Avg Precision: {ap:.4f}")
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))


# ── SHAP explainer ────────────────────────────────────────────────────────────

def build_explainer(model, X_train_sample, feature_names):
    print("[shap] Building TreeExplainer …")
    # Background sample of 500 rows keeps inference-time SHAP calls fast
    bg = shap.sample(X_train_sample, 500, random_state=42)
    explainer = shap.TreeExplainer(model, bg)
    print("[shap] Done.")
    return explainer


# ── Persist artifacts ─────────────────────────────────────────────────────────

def save_artifacts(model, explainer, feature_names, threshold):
    model_path = os.path.join(MODEL_DIR, "xgb_fraud.json")
    model.save_model(model_path)

    exp_path = os.path.join(MODEL_DIR, "explainer.pkl")
    with open(exp_path, "wb") as f:
        pickle.dump(explainer, f)

    feat_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)

    thresh_path = os.path.join(MODEL_DIR, "threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"threshold": threshold}, f)

    print(f"\n[train] Artifacts saved to {MODEL_DIR}/")
    print(f"  → {model_path}")
    print(f"  → {exp_path}")
    print(f"  → {feat_path}")
    print(f"  → {thresh_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/creditcard.csv")
    args = parser.parse_args()

    df = load_data(args.data)
    X, y, feature_names = preprocess(df)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    print(f"[train] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    model = train(X_train, y_train, X_val, y_val)
    threshold = tune_threshold(model, X_val, y_val)
    evaluate(model, X_test, y_test, threshold)

    explainer = build_explainer(model, X_train, feature_names)
    save_artifacts(model, explainer, feature_names, threshold)
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
