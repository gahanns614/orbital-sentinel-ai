"""
ORBITAL SENTINEL AI — scenario_runner.py
Day 2 scope: run the simulator and inject one or more attacks on a
timeline, so we can demo single attacks AND multi-stage/simultaneous
attacks (the differentiator called out in the plan).

Run examples:
    # single attack, starts immediately, runs 15s
    python simulator/scenario_runner.py --attack signal_jamming --duration 15

    # multi-stage: jamming at t=0, brute force joins at t=5
    python simulator/scenario_runner.py --scenario kill_chain_demo

    # list available attacks
    python simulator/scenario_runner.py --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from satellite_sim import SatelliteSimulator
from attack_engine import ATTACK_REGISTRY


# Predefined multi-stage scenarios: list of (start_offset_seconds, attack_name)
SCENARIOS = {
    "kill_chain_demo": [
        (0.0, "signal_jamming"),
        (5.0, "brute_force"),
    ],
    "spoof_and_replay": [
        (0.0, "signal_spoofing"),
        (3.0, "replay_attack"),
    ],
    "full_assault": [
        (0.0, "ddos"),
        (2.0, "brute_force"),
        (6.0, "signal_jamming"),
    ],
}


class ActiveAttack:
    def __init__(self, attack_instance, start_time):
        self.attack = attack_instance
        self.start_time = start_time


def run(hz: float, duration: float, single_attack: str = None, scenario: str = None):
    sim = SatelliteSimulator()
    interval = 1.0 / hz
    t0 = time.time()

    # Build the attack timeline
    timeline = []  # list of (offset_seconds, attack_name)
    if scenario:
        if scenario not in SCENARIOS:
            print(f"Unknown scenario '{scenario}'. Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        timeline = SCENARIOS[scenario]
    elif single_attack:
        if single_attack not in ATTACK_REGISTRY:
            print(f"Unknown attack '{single_attack}'. Available: {list(ATTACK_REGISTRY.keys())}")
            sys.exit(1)
        timeline = [(0.0, single_attack)]

    active_attacks = []  # list of ActiveAttack
    pending = sorted(timeline, key=lambda x: x[0])

    label = scenario or single_attack or "normal-only"
    print(f"[orbital-sentinel] scenario_runner starting — mode='{label}', {hz} Hz, {duration}s", file=sys.stderr)

    while (time.time() - t0) < duration:
        now = time.time() - t0

        # activate any attacks whose start time has arrived
        while pending and pending[0][0] <= now:
            offset, attack_name = pending.pop(0)
            attack_cls = ATTACK_REGISTRY[attack_name]
            active_attacks.append(ActiveAttack(attack_cls(), now))
            print(f"[orbital-sentinel] ATTACK STARTED: {attack_name} at t={now:.1f}s", file=sys.stderr)

        frame = sim.next_frame()

        # apply every currently active attack, in order (simultaneous
        # attacks compose -- later attacks mutate the already-mutated frame)
        for aa in active_attacks:
            frame = aa.attack.apply(frame, now - aa.start_time)

        print(json.dumps(frame))
        time.sleep(interval)

    print("[orbital-sentinel] scenario complete.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Orbital Sentinel AI — scenario runner")
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--attack", type=str, default=None, help="Run a single attack from t=0")
    parser.add_argument("--scenario", type=str, default=None, help="Run a predefined multi-stage scenario")
    parser.add_argument("--list", action="store_true", help="List available attacks and scenarios")
    args = parser.parse_args()

    if args.list:
        print("Available attacks:", list(ATTACK_REGISTRY.keys()))
        print("Available scenarios:", list(SCENARIOS.keys()))
        return

    run(args.hz, args.duration, single_attack=args.attack, scenario=args.scenario)


if __name__ == "__main__":
    main()
