"""
ORBITAL SENTINEL AI — ml/models/attack_classifier.py
Day 11 scope: multi-class attack classifier.

POSITIONING (per the ML architecture research): this is the PRIMARY
accuracy-driving detector, not the Isolation Forest. Research strongly
supports XGBoost/tree-ensemble classifiers for exactly this kind of
tabular multi-class attack classification (near-perfect F1 on SDN/IoT
intrusion benchmarks). Isolation Forest's job (previous file) is fast
first-pass filtering + catching unknown/zero-day anomalies; THIS model's
job is accurately naming which known attack is occurring.

Falls back to sklearn's GradientBoostingClassifier if xgboost isn't
installed -- not every teammate's machine will have it set up, and the
fallback is a legitimate, literature-supported alternative, not a
placeholder.

Run:
    python ml/models/attack_classifier.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_dataset import FLAT_FIELDS

REGISTRY_DIR = Path(__file__).parent.parent / "registry"

try:
    from xgboost import XGBClassifier
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    BACKEND = "sklearn_gradient_boosting"


def build_model():
    if BACKEND == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )


def train_and_evaluate(df: pd.DataFrame):
    X = df[FLAT_FIELDS].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    # time-respecting-ish split: still random here since each class was
    # generated independently (no cross-class time leakage possible),
    # but stratified to keep class balance in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[attack_classifier] backend: {BACKEND}")
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Attack classifier evaluation (held-out 20% test set) ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print("Classes:", list(le.classes_))
    print(cm)

    # feature importance -- useful for the Devpost "how we built it"
    # explainability story
    if hasattr(model, "feature_importances_"):
        importances = sorted(zip(FLAT_FIELDS, model.feature_importances_), key=lambda x: -x[1])
        print("\nTop 8 most important features:")
        for feat, imp in importances[:8]:
            print(f"  {feat:30s} {imp:.4f}")

    return model, le


def main():
    dataset_path = Path(__file__).parent.parent / "data" / "telemetry_dataset.csv"
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Run ml/generate_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"[attack_classifier] loaded {len(df)} rows, {df['label'].nunique()} classes")

    model, label_encoder = train_and_evaluate(df)

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, REGISTRY_DIR / "attack_classifier.joblib")
    joblib.dump(label_encoder, REGISTRY_DIR / "label_encoder.joblib")
    print(f"\n[attack_classifier] saved model + label encoder to {REGISTRY_DIR}")


if __name__ == "__main__":
    main()
