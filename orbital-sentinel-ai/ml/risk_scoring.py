"""
ORBITAL SENTINEL AI — ml/risk_scoring.py
Day 13 scope: fuse the outputs of all 3 ML models into ONE risk score.

DESIGN PRINCIPLE (from the original blueprint, worth restating): this is
DELIBERATELY a deterministic weighted formula, not a 4th trained model.
A black-box risk score is a demo liability -- when a judge asks "why is
this 87 and not 60?", "here's the formula" is a far stronger answer than
"the model said so." Every component of the score is inspectable.

risk_score = f(anomaly_signal, classifier_confidence x attack_severity,
                failure_probability)

Severity weights are pulled directly from each Attack class's `severity`
attribute already defined in simulator/attack_engine/ -- not redefined
here, so there's exactly one source of truth for "how bad is this attack."

Run:
    python ml/risk_scoring.py     # demo: scores a few example scenarios
"""

import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from generate_dataset import FLAT_FIELDS
from generate_timeseries_dataset import COMMS_FEATURES, WINDOW_SIZE
from attack_engine import ATTACK_REGISTRY

REGISTRY_DIR = Path(__file__).parent / "registry"

# Pull severity directly from the attack classes -- single source of
# truth, no magic numbers duplicated here.
ATTACK_SEVERITY = {name: cls.severity for name, cls in ATTACK_REGISTRY.items()}
ATTACK_SEVERITY["normal"] = 0.0
ATTACK_SEVERITY["anomaly"] = 0.5  # unknown/unclassified anomaly -- moderate default

# Weights for the fusion formula. Kept as named constants (not buried
# magic numbers) so they're easy to tune live if a demo run needs it.
WEIGHT_ANOMALY = 0.25
WEIGHT_CLASSIFIER = 0.45
WEIGHT_FAILURE_PREDICTION = 0.30

RISK_THRESHOLDS = [(80, "CRITICAL"), (55, "HIGH"), (25, "MEDIUM"), (0, "LOW")]


def risk_level_for_score(score: float) -> str:
    for threshold, level in RISK_THRESHOLDS:
        if score >= threshold:
            return level
    return "LOW"


class RiskScorer:
    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.anomaly_model = joblib.load(registry_dir / "isolation_forest.joblib")
        self.classifier = joblib.load(registry_dir / "attack_classifier.joblib")
        self.label_encoder = joblib.load(registry_dir / "label_encoder.joblib")
        self.failure_norm = joblib.load(registry_dir / "failure_predictor_norm.joblib")

        lstm_path = registry_dir / "failure_predictor_lstm.pt"
        gb_path = registry_dir / "failure_predictor_gb.joblib"
        if lstm_path.exists():
            import torch
            from models.failure_predictor import FailureLSTM
            self.failure_backend = "pytorch_lstm"
            self.failure_model = FailureLSTM(n_features=len(COMMS_FEATURES))
            self.failure_model.load_state_dict(torch.load(lstm_path))
            self.failure_model.eval()
        elif gb_path.exists():
            self.failure_backend = "sklearn_gradient_boosting"
            self.failure_model = joblib.load(gb_path)
        else:
            raise FileNotFoundError("No failure predictor model found in registry.")

        # rolling window buffer of recent comms features, needed by the
        # failure predictor which looks at a WINDOW_SIZE-tick history
        self.comms_history = deque(maxlen=WINDOW_SIZE)

    def _predict_failure_probability(self) -> float:
        """Returns None if we don't have enough history yet, otherwise a
        0-1 probability of comms failure within the trained HORIZON."""
        if len(self.comms_history) < WINDOW_SIZE:
            return None

        window = np.array(self.comms_history)  # shape (WINDOW_SIZE, n_features)
        mean, std = self.failure_norm["mean"], self.failure_norm["std"]
        window_norm = (window - mean) / std

        if self.failure_backend == "pytorch_lstm":
            import torch
            with torch.no_grad():
                x = torch.tensor(window_norm[None, :, :], dtype=torch.float32)
                prob = self.failure_model(x).item()
        else:
            flat = window_norm.reshape(1, -1)
            prob = self.failure_model.predict_proba(flat)[0, 1]

        return float(prob)

    def score(self, frame: dict) -> dict:
        """Score one telemetry frame. Call this once per incoming frame
        in sequence (it maintains rolling history internally for the
        failure predictor)."""
        # --- flatten frame into the same feature layout used at training time ---
        seq = frame["command_stream"]["seq"]
        prev_seq = getattr(self, "_prev_seq", None)
        auth_token_is_valid = 1 if frame["command_stream"]["auth_token"] == "tok_valid_demo" else 0
        row = {
            "orbit_lat": frame["orbit"]["lat"], "orbit_lon": frame["orbit"]["lon"],
            "orbit_alt_km": frame["orbit"]["alt_km"],
            "power_battery_pct": frame["power"]["battery_pct"],
            "power_solar_efficiency": frame["power"]["solar_efficiency"],
            "power_fuel_pct": frame["power"]["fuel_pct"],
            "thermal_temp_c": frame["thermal"]["temp_c"],
            "compute_cpu_pct": frame["compute"]["cpu_pct"], "compute_mem_pct": frame["compute"]["mem_pct"],
            "comms_signal_dbm": frame["comms"]["signal_dbm"], "comms_snr_db": frame["comms"]["snr_db"],
            "comms_noise_floor_db": frame["comms"]["noise_floor_db"],
            "comms_frequency_mhz": frame["comms"]["frequency_mhz"],
            "comms_latency_ms": frame["comms"]["latency_ms"],
            "comms_packet_loss_pct": frame["comms"]["packet_loss_pct"],
            "command_seq_delta": seq - prev_seq if prev_seq is not None else 1,
            "auth_token_is_valid": auth_token_is_valid,
            "security_auth_failures_last_min": frame["security"]["auth_failures_last_min"],
            "security_active_sessions": frame["security"]["active_sessions"],
        }
        self._prev_seq = seq
        X = np.array([[row[f] for f in FLAT_FIELDS]])

        # --- component 1: anomaly signal (Isolation Forest) ---
        anomaly_raw = -self.anomaly_model.decision_function(X)[0]  # higher = more anomalous
        anomaly_signal = float(np.clip((anomaly_raw + 0.1) / 0.3, 0, 1))  # rough 0-1 normalization

        # --- component 2: classifier confidence x attack severity ---
        proba = self.classifier.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        predicted_label = self.label_encoder.classes_[pred_idx]
        confidence = float(proba[pred_idx])
        severity = ATTACK_SEVERITY.get(predicted_label, 0.5)
        classifier_signal = confidence * severity if predicted_label != "normal" else 0.0

        # --- component 3: failure prediction (needs rolling history) ---
        self.comms_history.append([frame["comms"][f] for f in COMMS_FEATURES])
        failure_probability = self._predict_failure_probability()
        failure_signal = failure_probability if failure_probability is not None else 0.0

        # --- fusion: deterministic weighted sum, scaled to 0-100 ---
        risk_score = 100 * (
            WEIGHT_ANOMALY * anomaly_signal +
            WEIGHT_CLASSIFIER * classifier_signal +
            WEIGHT_FAILURE_PREDICTION * failure_signal
        )
        risk_score = float(np.clip(risk_score, 0, 100))

        return {
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level_for_score(risk_score),
            "predicted_attack": predicted_label,
            "classifier_confidence": round(confidence, 3),
            "anomaly_signal": round(anomaly_signal, 3),
            "failure_probability": round(failure_probability, 3) if failure_probability is not None else None,
            "breakdown": {
                "anomaly_contribution": round(100 * WEIGHT_ANOMALY * anomaly_signal, 1),
                "classifier_contribution": round(100 * WEIGHT_CLASSIFIER * classifier_signal, 1),
                "failure_contribution": round(100 * WEIGHT_FAILURE_PREDICTION * failure_signal, 1),
            },
        }


def main():
    """Demo: run a few example scenarios through the fused scorer to
    prove the risk score behaves sensibly."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
    from satellite_sim import SatelliteSimulator

    scorer = RiskScorer()

    scenarios = [
        ("Normal operation", None),
        ("Signal jamming", "signal_jamming"),
        ("Brute force", "brute_force"),
        ("Gradual comm degradation (mid-ramp)", "comm_link_degradation"),
    ]

    for name, attack_name in scenarios:
        sim = SatelliteSimulator()
        attack = ATTACK_REGISTRY[attack_name]() if attack_name else None
        print(f"\n=== {name} ===")
        # feed enough frames to fill the rolling window and show the fused score stabilize
        for i in range(WINDOW_SIZE + 3):
            frame = sim.next_frame()
            if attack is not None:
                t = i * 8.0  # push well into the ramp so the effect is visible
                frame = attack.apply(frame, t)
            result = scorer.score(frame)
        print(f"  Risk Score: {result['risk_score']} ({result['risk_level']})")
        print(f"  Predicted:  {result['predicted_attack']} (confidence {result['classifier_confidence']})")
        print(f"  Failure probability (next {WINDOW_SIZE + 5}s): {result['failure_probability']}")
        print(f"  Breakdown: {result['breakdown']}")


if __name__ == "__main__":
    main()
