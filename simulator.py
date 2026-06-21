"""
simulator.py — Generates a live stream of synthetic transactions

Mimics the Kaggle credit card dataset's feature distribution so the
trained model can score these transactions the same way it scores
real data:
    - V1–V28: PCA components, approximately standard-normal
    - Amount: log-normal for legit transactions, more uniform for fraud
    - Time_scaled: placeholder (not informative in the trained model)

Fraud transactions get a deliberate shift applied to the V-features
that mirrors the most fraud-discriminative columns in the real
dataset (V14, V12, V10, V17, V3, V1 tend to separate fraud from legit
most strongly). This isn't meant to be statistically perfect — it's
meant to produce a believable, demo-able live stream.

Used by api.py's /stream SSE endpoint.
"""

import random
import time
from typing import Generator

import numpy as np

# Rough per-feature means/stds estimated from the Kaggle dataset.
# V1–V28 are already PCA-normalized, so most hover near N(0, ~1.8).
FEATURE_MEANS = {
    "V1": -0.3, "V2": 0.5, "V3": -0.8, "V4": 0.6,
    "V5": -0.2, "V6": 0.1, "V7": -0.3, "V8": 0.1,
    "V9": -0.4, "V10": -0.5, "V11": 0.4, "V12": -0.6,
    "V13": 0.0, "V14": -0.7, "V15": 0.1, "V16": -0.4,
    "V17": -0.3, "V18": -0.2, "V19": 0.1, "V20": 0.1,
    "V21": 0.1, "V22": 0.0, "V23": 0.0, "V24": 0.0,
    "V25": 0.1, "V26": 0.0, "V27": 0.0, "V28": 0.0,
}

FEATURE_STDS = {k: 1.8 for k in FEATURE_MEANS}

# Shift applied to fraud transactions on the most fraud-discriminative
# features, so the trained model actually has a believable signal to find.
FRAUD_DELTAS = {
    "V1": -4.5, "V2": 3.2, "V3": -5.0, "V4": 3.0,
    "V10": -5.0, "V11": 3.0, "V12": -6.0, "V14": -8.0,
    "V16": -3.5, "V17": -6.0,
}

LEGIT_MERCHANTS = [
    "Whole Foods", "Amazon", "Netflix", "Starbucks", "Uber",
    "Target", "Spotify", "Apple Store", "Delta Airlines", "Marriott",
    "Shell Gas", "Walmart", "CVS Pharmacy", "Home Depot", "Best Buy",
]

# Mostly nonsense/shell-like names — occasionally a legit-looking
# merchant gets used too, since real fraud doesn't always look exotic.
FRAUD_MERCHANTS = [
    "Unknown Vendor #4821", "FX_CONVERT_SVC", "CryptoExch-EU",
    "INTL_WIRE_0039", "AnonPay LLC", "MicroTx Services",
]


def _random_merchant(is_fraud: bool) -> str:
    if is_fraud:
        return random.choice(FRAUD_MERCHANTS) if random.random() < 0.6 else random.choice(LEGIT_MERCHANTS)
    return random.choice(LEGIT_MERCHANTS)


def _random_transaction(is_fraud: bool, tx_id: int) -> dict:
    features = {}
    for name, mean in FEATURE_MEANS.items():
        std = FEATURE_STDS[name]
        delta = FRAUD_DELTAS.get(name, 0) if is_fraud else 0
        features[name] = round(float(np.random.normal(mean + delta, std)), 6)

    if is_fraud:
        amount = round(float(np.random.uniform(1, 500)), 2)
    else:
        amount = round(float(np.random.lognormal(3.5, 1.5)), 2)

    # Approximate the same scaling train.py applies to Amount/Time
    features["Amount_scaled"] = round((amount - 88.35) / 250.12, 6)
    features["Time_scaled"] = 0.0

    return {
        "id": tx_id,
        "amount": amount,
        "merchant": _random_merchant(is_fraud),
        "features": features,
        "true_label": int(is_fraud),  # ground truth, for demo accuracy display only
    }


def transaction_stream(
    fraud_rate: float = 0.015,
    delay_seconds: float = 1.2,
) -> Generator[dict, None, None]:
    """
    Infinite generator of simulated transactions.

    Args:
        fraud_rate: probability any given transaction is fraudulent.
        delay_seconds: base delay between transactions (jittered ±0.3–0.5s
            to feel like a real stream rather than a metronome).
    """
    tx_id = 1
    while True:
        is_fraud = random.random() < fraud_rate
        yield _random_transaction(is_fraud, tx_id)
        tx_id += 1
        time.sleep(max(0.1, delay_seconds + random.uniform(-0.3, 0.5)))


if __name__ == "__main__":
    # Quick manual smoke test: print 10 transactions and exit.
    stream = transaction_stream(fraud_rate=0.3, delay_seconds=0.1)
    for i, tx in enumerate(stream):
        label = "FRAUD" if tx["true_label"] else "legit"
        print(f"[{label}] #{tx['id']} {tx['merchant']} — ${tx['amount']}")
        if i >= 9:
            break
