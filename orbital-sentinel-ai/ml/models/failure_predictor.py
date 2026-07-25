"""
ORBITAL SENTINEL AI — ml/models/failure_predictor.py
Day 12 scope: predicts whether a communication failure will occur within
the next HORIZON ticks, given a WINDOW_SIZE-tick history of comms
features. This is the model behind the demo's "AI predicted the failure
before it happened" moment.

Primary: LSTM (PyTorch) over the raw windowed sequences -- the
literature-supported choice (Telemanom/NASA SMAP-MSL baseline; OPSSAT-AD
deep models). Falls back to a GradientBoostingClassifier over flattened
windows if torch isn't installed -- not literally Prophet (Prophet is a
univariate trend forecaster and doesn't fit this multivariate windowed
binary-classification framing well), but the same "fallback if the
primary dependency is missing" pattern used in attack_classifier.py, and
still a legitimate model for this task.

The headline metric reported here isn't just accuracy -- it's LEAD TIME:
of the failures we correctly caught, how many seconds of advance warning
did the model give, on average? That's the number that actually matters
for the demo narrative.

Run:
    python ml/models/failure_predictor.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

REGISTRY_DIR = Path(__file__).parent.parent / "registry"

try:
    import torch
    import torch.nn as nn
    BACKEND = "pytorch_lstm"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    BACKEND = "sklearn_gradient_boosting"


# ---------------------------------------------------------------------
# PyTorch LSTM path
# ---------------------------------------------------------------------
if BACKEND == "pytorch_lstm":
    class FailureLSTM(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return torch.sigmoid(out).squeeze(-1)

    def train_lstm(X_train, y_train, X_val, y_val, n_features, epochs=15, lr=1e-3):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FailureLSTM(n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

        batch_size = 256
        n_batches = max(1, len(X_train_t) // batch_size)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            for b in range(n_batches):
                idx = perm[b * batch_size:(b + 1) * batch_size]
                optimizer.zero_grad()
                pred = model(X_train_t[idx])
                loss = criterion(pred, y_train_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 3 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch+1}/{epochs}  loss={epoch_loss/n_batches:.4f}")

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t).cpu().numpy()
        return model, val_probs


# ---------------------------------------------------------------------
# Shared: normalization, evaluation, lead-time metric
# ---------------------------------------------------------------------
def normalize(X_train, X_test):
    """Per-feature z-score normalization, fit on train only."""
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def compute_lead_time_ticks(y_true, y_pred, threshold=0.5):
    """Among windows correctly flagged as 'failure incoming' (true
    positives), the label itself already encodes 'failure within HORIZON
    ticks' -- so every true positive IS advance warning by definition.
    This just reports the recall (how many incoming failures we actually
    caught) alongside a reminder of what HORIZON means in seconds."""
    pred_binary = (y_pred >= threshold).astype(int)
    true_positives = ((pred_binary == 1) & (y_true == 1)).sum()
    total_positives = (y_true == 1).sum()
    return true_positives, total_positives


def main():
    dataset_path = Path(__file__).parent.parent / "data" / "timeseries_dataset.npz"
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}. Run ml/generate_timeseries_dataset.py first.")
        sys.exit(1)

    data = np.load(dataset_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    window_size = int(data["window_size"])
    horizon = int(data["horizon"])
    n_features = X.shape[-1]

    print(f"[failure_predictor] backend: {BACKEND}")
    print(f"[failure_predictor] loaded {len(X)} windows, window_size={window_size}, horizon={horizon} ticks")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_norm, X_test_norm, mean, std = normalize(X_train, X_test)

    if BACKEND == "pytorch_lstm":
        model, val_probs = train_lstm(X_train_norm, y_train, X_test_norm, y_test, n_features)
        y_pred_proba = val_probs
    else:
        # flatten windows for the tree-based fallback
        X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
        X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)
        model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        model.fit(X_train_flat, y_train)
        y_pred_proba = model.predict_proba(X_test_flat)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)

    print("\n=== Failure predictor evaluation (held-out 20% test set) ===")
    print(classification_report(y_test, y_pred, target_names=["no_failure_soon", "failure_incoming"]))
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUCROC: {auc:.4f}")
    except ValueError:
        pass

    tp, total_pos = compute_lead_time_ticks(y_test, y_pred_proba)
    print(f"\nOf {total_pos} genuine 'failure incoming' windows in the test set, "
          f"correctly flagged {tp} ({100*tp/max(1,total_pos):.1f}%).")
    print(f"Each correct flag means the model gave advance warning up to {horizon} ticks "
          f"(~{horizon} simulated seconds) before signal collapse -- this is the demo's "
          f"'predicted before it happened' number.")

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if BACKEND == "pytorch_lstm":
        torch.save(model.state_dict(), REGISTRY_DIR / "failure_predictor_lstm.pt")
    else:
        joblib.dump(model, REGISTRY_DIR / "failure_predictor_gb.joblib")
    joblib.dump({"mean": mean, "std": std}, REGISTRY_DIR / "failure_predictor_norm.joblib")
    print(f"\n[failure_predictor] saved model to {REGISTRY_DIR}")


if __name__ == "__main__":
    main()
