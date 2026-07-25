"""
ORBITAL SENTINEL AI — event_publisher.py
Day 3-4 scope: publish telemetry frames onto a Redis Stream so the
simulator and everything downstream (ML inference, backend) are properly
decoupled -- they no longer need to be the same process.

Requires: pip install redis
Requires: a Redis-compatible server running on localhost:6379
    (Memurai on Windows, or real Redis via Docker/apt on Mac/Linux)

Run:
    python simulator/event_publisher.py --scenario kill_chain_demo --duration 15
    python simulator/event_publisher.py --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from satellite_sim import SatelliteSimulator
from attack_engine import ATTACK_REGISTRY
from scenario_runner import SCENARIOS, ActiveAttack

try:
    import redis
except ImportError:
    print("ERROR: the 'redis' package isn't installed. Run: pip install redis")
    sys.exit(1)

STREAM_NAME = "telemetry_frames"


def get_redis_client():
    try:
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        return client
    except redis.exceptions.ConnectionError as e:
        print(f"ERROR: could not connect to Redis/Memurai on localhost:6379 -- {e}")
        print("Make sure Memurai (or Redis) is running, then try again.")
        sys.exit(1)


def run(hz: float, duration: float, single_attack: str = None, scenario: str = None):
    r = get_redis_client()
    print(f"[event_publisher] connected to Redis. Publishing to stream '{STREAM_NAME}'.", file=sys.stderr)

    sim = SatelliteSimulator()
    interval = 1.0 / hz
    t0 = time.time()

    timeline = []
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

    active_attacks = []
    pending = sorted(timeline, key=lambda x: x[0])
    label = scenario or single_attack or "normal-only"
    print(f"[event_publisher] mode='{label}', {hz} Hz, {duration}s", file=sys.stderr)

    frame_count = 0
    while (time.time() - t0) < duration:
        now = time.time() - t0

        while pending and pending[0][0] <= now:
            offset, attack_name = pending.pop(0)
            attack_cls = ATTACK_REGISTRY[attack_name]
            active_attacks.append(ActiveAttack(attack_cls(), now))
            print(f"[event_publisher] ATTACK STARTED: {attack_name} at t={now:.1f}s", file=sys.stderr)

        frame = sim.next_frame()
        for aa in active_attacks:
            frame = aa.attack.apply(frame, now - aa.start_time)

        # XADD requires a flat dict of field->value strings; we nest the
        # full frame as a single JSON field rather than flattening every
        # subfield, since consumers just need to json.loads() it back.
        r.xadd(STREAM_NAME, {"data": json.dumps(frame)}, maxlen=10000, approximate=True)
        frame_count += 1
        time.sleep(interval)

    print(f"[event_publisher] done. Published {frame_count} frames to '{STREAM_NAME}'.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Orbital Sentinel AI — Redis Stream publisher")
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available attacks:", list(ATTACK_REGISTRY.keys()))
        print("Available scenarios:", list(SCENARIOS.keys()))
        return

    run(args.hz, args.duration, single_attack=args.attack, scenario=args.scenario)


if __name__ == "__main__":
    main()
