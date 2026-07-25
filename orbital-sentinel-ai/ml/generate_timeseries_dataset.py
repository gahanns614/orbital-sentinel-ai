"""
ORBITAL SENTINEL AI — ml/generate_timeseries_dataset.py
Day 12 scope: generate windowed time-series data for the failure
predictor. Unlike generate_dataset.py (single-frame snapshots for the
classifier), this generates SESSIONS -- sequences of frames over
simulated time -- because predicting an upcoming failure requires seeing
a trend, not a single moment.

Each session is a synthetic "session_length" tick timeline. In HALF the
sessions, comm_link_degradation starts at a random point and decays
toward failure; in the other half, the link stays normal throughout.
Ticks are treated as 1-simulated-second each for label purposes (not
real wall-clock time -- generation is instant, not tied to sleep()).

LABELING: for every tick, label = 1 if a failure (comms_signal_dbm <
FAILURE_SIGNAL_THRESHOLD) occurs anywhere in the next HORIZON ticks,
else 0. This is what lets the model learn to say "failure incoming"
BEFORE the signal actually collapses, which is the whole point.

Run:
    python ml/generate_timeseries_dataset.py
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from satellite_sim import SatelliteSimulator
from attack_engine import CommLinkDegradation

WINDOW_SIZE = 10       # ticks of history the model sees
HORIZON = 15            # ticks ahead we're predicting failure within
FAILURE_SIGNAL_THRESHOLD = -105.0  # dBm below which we call it "failed"
SESSION_LENGTH = 180     # ticks per session
COMMS_FEATURES = ["signal_dbm", "snr_db", "noise_floor_db", "latency_ms", "packet_loss_pct"]


def run_session(will_degrade: bool, session_length: int = SESSION_LENGTH):
    """Generate one session's comms feature history, tick by tick."""
    sim = SatelliteSimulator()
    attack = None
    attack_start_tick = None

    if will_degrade:
        attack = CommLinkDegradation(ramp_seconds=90.0)
        # start the degradation at a random point so the model sees
        # varied amounts of "lead-up" data, not always the same pattern
        attack_start_tick = np.random.randint(20, session_length - 60)

    history = []
    for tick in range(session_length):
        frame = sim.next_frame()
        if attack is not None and tick >= attack_start_tick:
            frame = attack.apply(frame, tick - attack_start_tick)
        history.append([frame["comms"][f] for f in COMMS_FEATURES])

    return np.array(history)  # shape: (session_length, num_features)


def label_session(history: np.ndarray) -> np.ndarray:
    """For each tick, label=1 if signal drops below threshold anywhere
    in the next HORIZON ticks."""
    signal = history[:, COMMS_FEATURES.index("signal_dbm")]
    n = len(signal)
    labels = np.zeros(n, dtype=int)
    for t in range(n):
        window_end = min(n, t + HORIZON)
        if np.any(signal[t:window_end] < FAILURE_SIGNAL_THRESHOLD):
            labels[t] = 1
    return labels


def build_windows(history: np.ndarray, labels: np.ndarray):
    """Slide a WINDOW_SIZE-tick window across the session. Each window's
    label is the label AT THE LAST TICK of the window (i.e. "given what
    I've seen up to now, will failure happen within HORIZON ticks?")."""
    X, y = [], []
    for t in range(WINDOW_SIZE, len(history)):
        X.append(history[t - WINDOW_SIZE:t])
        y.append(labels[t])
    return np.array(X), np.array(y)


def main():
    n_sessions = 150  # 75 degrading + 75 normal-only
    all_X, all_y = [], []

    print(f"[generate_timeseries_dataset] generating {n_sessions} sessions "
          f"({SESSION_LENGTH} ticks each, window={WINDOW_SIZE}, horizon={HORIZON})...")

    for i in range(n_sessions):
        will_degrade = (i % 2 == 0)
        history = run_session(will_degrade)
        labels = label_session(history)
        X, y = build_windows(history, labels)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "timeseries_dataset.npz",
        X=X, y=y,
        feature_names=np.array(COMMS_FEATURES),
        window_size=WINDOW_SIZE, horizon=HORIZON,
        failure_threshold=FAILURE_SIGNAL_THRESHOLD,
    )

    print(f"[generate_timeseries_dataset] built {len(X)} windows, "
          f"shape {X.shape}, positive rate {y.mean():.2%}")
    print(f"[generate_timeseries_dataset] saved to {output_dir / 'timeseries_dataset.npz'}")


if __name__ == "__main__":
    main()
