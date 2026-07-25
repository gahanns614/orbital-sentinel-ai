"""
ORBITAL SENTINEL AI — ml/generate_dataset.py
Day 9 scope: generate a labeled synthetic dataset for training the
anomaly detector and attack classifier.

SCOPING DECISION: each class (normal + 5 attacks) is generated
independently with a clean single label, rather than reusing the
multi-stage scenario_runner.py logic. This deliberately sidesteps the
known attack_label limitation (it only stores the most-recently-applied
attack during simultaneous/multi-stage attacks -- flagged back on Day 2)
by not generating multi-attack frames for this first training pass.
Multi-label training on simultaneous attacks is a legitimate stretch
goal, not something to build under hackathon time pressure right now.

Ground truth is exact by construction: we know the label because WE
applied the attack, not because we inferred it -- this is the standard
approach when no public labeled dataset exists for a domain.

Run:
    python ml/generate_dataset.py
    python ml/generate_dataset.py --samples-per-class 3000 --output ml/data/telemetry_dataset.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from satellite_sim import SatelliteSimulator
from attack_engine import ATTACK_REGISTRY

FLAT_FIELDS = [
    "orbit_lat", "orbit_lon", "orbit_alt_km",
    "power_battery_pct", "power_solar_efficiency", "power_fuel_pct",
    "thermal_temp_c",
    "compute_cpu_pct", "compute_mem_pct",
    "comms_signal_dbm", "comms_snr_db", "comms_noise_floor_db",
    "comms_frequency_mhz", "comms_latency_ms", "comms_packet_loss_pct",
    "command_seq_delta",     # engineered: change in seq vs previous frame (replay detection signal)
    "auth_token_is_valid",   # engineered: 1 if auth_token matches the known-good baseline, else 0 (spoofing detection signal)
    "security_auth_failures_last_min", "security_active_sessions",
]


def flatten_frame(frame: dict, prev_seq: int) -> dict:
    """Flatten the nested schema into a flat feature row. prev_seq lets us
    compute seq_delta, which is the actual replay-attack tell (real seq
    increases; replay repeats/goes backward)."""
    seq = frame["command_stream"]["seq"]
    auth_token_is_valid = 1 if frame["command_stream"]["auth_token"] == "tok_valid_demo" else 0
    return {
        "orbit_lat": frame["orbit"]["lat"],
        "orbit_lon": frame["orbit"]["lon"],
        "orbit_alt_km": frame["orbit"]["alt_km"],
        "power_battery_pct": frame["power"]["battery_pct"],
        "power_solar_efficiency": frame["power"]["solar_efficiency"],
        "power_fuel_pct": frame["power"]["fuel_pct"],
        "thermal_temp_c": frame["thermal"]["temp_c"],
        "compute_cpu_pct": frame["compute"]["cpu_pct"],
        "compute_mem_pct": frame["compute"]["mem_pct"],
        "comms_signal_dbm": frame["comms"]["signal_dbm"],
        "comms_snr_db": frame["comms"]["snr_db"],
        "comms_noise_floor_db": frame["comms"]["noise_floor_db"],
        "comms_frequency_mhz": frame["comms"]["frequency_mhz"],
        "comms_latency_ms": frame["comms"]["latency_ms"],
        "comms_packet_loss_pct": frame["comms"]["packet_loss_pct"],
        "command_seq_delta": seq - prev_seq if prev_seq is not None else 1,
        "auth_token_is_valid": auth_token_is_valid,
        "security_auth_failures_last_min": frame["security"]["auth_failures_last_min"],
        "security_active_sessions": frame["security"]["active_sessions"],
    }, seq


def generate_class(label: str, n_samples: int, attack_name: str = None):
    """Generate n_samples frames for one class (normal, or a specific attack).
    Each class gets its own fresh simulator instance so classes don't
    contaminate each other's orbital/telemetry state."""
    sim = SatelliteSimulator()
    attack = ATTACK_REGISTRY[attack_name]() if attack_name else None
    rows = []
    prev_seq = None

    for i in range(n_samples):
        frame = sim.next_frame()
        if attack is not None:
            # ramp attacks (jamming, ddos) need a nonzero t_since_start to
            # show their full effect range across the generated samples,
            # not just the instant-start value
            t_since_start = (i / n_samples) * 15.0
            frame = attack.apply(frame, t_since_start)
        row, prev_seq = flatten_frame(frame, prev_seq)
        row["label"] = label
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate labeled training data for Orbital Sentinel AI")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--output", type=str, default="ml/data/telemetry_dataset.csv")
    args = parser.parse_args()

    classes = [("normal", None)] + [(name, name) for name in ATTACK_REGISTRY.keys()]

    all_rows = []
    for label, attack_name in classes:
        print(f"[generate_dataset] generating {args.samples_per_class} samples for class '{label}'...")
        all_rows.extend(generate_class(label, args.samples_per_class, attack_name))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FLAT_FIELDS + ["label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[generate_dataset] wrote {len(all_rows)} rows across {len(classes)} classes to {output_path}")
    print(f"[generate_dataset] classes: {[c[0] for c in classes]}")


if __name__ == "__main__":
    main()
