"""
ORBITAL SENTINEL AI — ml/models/anomaly_detector.py
Day 10-11 scope: Isolation Forest anomaly detector.

POSITIONING (per the ML architecture research): Isolation Forest is a
fast, cheap first-stage filter, NOT the primary accuracy driver. The
OPSSAT-AD benchmark (real ESA satellite telemetry) showed Isolation
Forest scoring only F1=0.295 in the universal-detector setting -- weak
compared to supervised methods. Its value here is speed (sub-millisecond
inference) and the fact that it needs NO labeled attack data to train,
only normal baseline data -- useful for catching unknown/zero-day
anomalies that were never in the attack catalog.

Trained ONLY on 'normal' class data (standard unsupervised anomaly
detection practice) -- it learns what NORMAL looks like, then flags
anything that deviates, whether or not that deviation matches a known
attack signature. This is what gives the project its "unknown/zero-day
anomaly" detection capability described in the blueprint's threat catalog.

Run:
    python ml/models/anomaly_detector.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_dataset import FLAT_FIELDS

REGISTRY_DIR = Path(__file__).parent.parent / "registry"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_anomaly_detector(df: pd.DataFrame, contamination: float = 0.05):
    """Train Isolation Forest on NORMAL data only. contamination is the
    expected proportion of outliers even within 'normal' baseline data
    (accounts for the natural random-walk noise in the simulator)."""
    normal_df = df[df["label"] == "normal"]
    X_train = normal_df[FLAT_FIELDS].values

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_anomaly_detector(model, df: pd.DataFrame):
    """Evaluate on the FULL dataset (normal + all attacks). Ground truth
    for this evaluation is binary: is_attack = (label != 'normal').
    This tells us how well a model trained on ONLY normal data can flag
    attacks it never saw labeled examples of -- the actual zero-day-style
    detection capability."""
    X = df[FLAT_FIELDS].values
    y_true = (df["label"] != "normal").astype(int).values

    # IsolationForest.predict returns 1 (inlier/normal) or -1 (outlier/anomaly)
    raw_pred = model.predict(X)
    y_pred = (raw_pred == -1).astype(int)

    # decision_function: higher = more normal, lower = more anomalous.
    # Flip sign so higher score = more anomalous, matching typical AUC convention.
    anomaly_scores = -model.decision_function(X)

    print("=== Isolation Forest evaluation (binary: normal vs any-attack) ===")
    print(classification_report(y_true, y_pred, target_names=["normal", "attack"]))
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
        print(f"AUCROC: {auc:.4f}")
    except ValueError:
        pass

    print("\nPer-class detection rate (recall) -- how often each attack type gets flagged:")
    for label in sorted(df["label"].unique()):
        if label == "normal":
            continue
        mask = df["label"] == label
        detected = (y_pred[mask.values] == 1).sum()
        total = mask.sum()
        print(f"  {label:20s}: {detected}/{total} flagged ({100*detected/total:.1f}%)")


def main():
    dataset_path = Path(__file__).parent.parent / "data" / "telemetry_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Run ml/generate_dataset.py first.")
        sys.exit(1)

    df = load_dataset(str(dataset_path))
    print(f"[anomaly_detector] loaded {len(df)} rows")

    model = train_anomaly_detector(df)
    evaluate_anomaly_detector(model, df)

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    model_path = REGISTRY_DIR / "isolation_forest.joblib"
    joblib.dump(model, model_path)
    print(f"\n[anomaly_detector] saved model to {model_path}")


if __name__ == "__main__":
    main()
