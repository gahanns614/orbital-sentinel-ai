"""
ORBITAL SENTINEL AI — satellite_sim.py
Day 1 scope: emit one NORMAL-mode telemetry frame per second, matching
data/schemas/telemetry_frame.schema.json exactly.

No Redis, no DB, no attacks yet — this script's only job today is to prove
the schema produces realistic-looking data. Attack injection, streaming,
and orbital propagation (sgp4) get layered on in later days per the plan.

Run:
    python simulator/satellite_sim.py
    python simulator/satellite_sim.py --hz 5 --duration 30
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone


class MeanRevertingWalk:
    """
    Ornstein-Uhlenbeck-style random walk: value drifts randomly but is
    pulled back toward a baseline. This is what makes telemetry look like
    real sensor data instead of obviously-fake pure Gaussian noise.
    """

    def __init__(self, baseline: float, volatility: float, reversion_speed: float = 0.15,
                 floor: float = None, ceiling: float = None):
        self.baseline = baseline
        self.value = baseline
        self.volatility = volatility
        self.reversion_speed = reversion_speed
        self.floor = floor
        self.ceiling = ceiling

    def step(self) -> float:
        pull = self.reversion_speed * (self.baseline - self.value)
        shock = random.gauss(0, self.volatility)
        self.value += pull + shock
        if self.floor is not None:
            self.value = max(self.floor, self.value)
        if self.ceiling is not None:
            self.value = min(self.ceiling, self.value)
        return round(self.value, 3)


class SatelliteSimulator:
    def __init__(self, satellite_id: str = "sentinel-1"):
        self.satellite_id = satellite_id
        self.seq = 10000

        # Orbit: simple placeholder walk for Day 1. Replaced with real
        # sgp4/skyfield Keplerian propagation once orbit accuracy matters.
        self.lat = MeanRevertingWalk(0, 0.8, floor=-90, ceiling=90)
        self.lon = MeanRevertingWalk(0, 1.2, floor=-180, ceiling=180)
        self.alt = MeanRevertingWalk(550, 0.5, floor=500, ceiling=600)

        self.battery = MeanRevertingWalk(87, 0.3, floor=0, ceiling=100)
        self.solar_eff = MeanRevertingWalk(0.91, 0.01, floor=0, ceiling=1)
        self.fuel = MeanRevertingWalk(76, 0.02, floor=0, ceiling=100)

        self.temp = MeanRevertingWalk(21, 0.4, floor=-40, ceiling=60)

        self.cpu = MeanRevertingWalk(34, 2.0, floor=0, ceiling=100)
        self.mem = MeanRevertingWalk(48, 1.5, floor=0, ceiling=100)

        self.signal = MeanRevertingWalk(-82, 1.0, floor=-120, ceiling=-40)
        self.snr = MeanRevertingWalk(14, 0.8, floor=0, ceiling=30)
        self.noise_floor = MeanRevertingWalk(-95, 0.5, floor=-130, ceiling=-60)
        self.frequency = MeanRevertingWalk(2245.5, 0.02, floor=2240, ceiling=2251)
        self.latency = MeanRevertingWalk(320, 8, floor=100, ceiling=1000)
        self.packet_loss = MeanRevertingWalk(0.4, 0.15, floor=0, ceiling=100)

    def next_frame(self) -> dict:
        self.seq += 1
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "satellite_id": self.satellite_id,
            "mode": "normal",
            "orbit": {
                "lat": self.lat.step(),
                "lon": self.lon.step(),
                "alt_km": self.alt.step(),
            },
            "power": {
                "battery_pct": self.battery.step(),
                "solar_efficiency": self.solar_eff.step(),
                "fuel_pct": self.fuel.step(),
            },
            "thermal": {
                "temp_c": self.temp.step(),
            },
            "compute": {
                "cpu_pct": self.cpu.step(),
                "mem_pct": self.mem.step(),
            },
            "comms": {
                "signal_dbm": self.signal.step(),
                "snr_db": self.snr.step(),
                "noise_floor_db": self.noise_floor.step(),
                "frequency_mhz": self.frequency.step(),
                "latency_ms": self.latency.step(),
                "packet_loss_pct": self.packet_loss.step(),
            },
            "command_stream": {
                "seq": self.seq,
                "checksum": f"{random.getrandbits(32):08x}",
                "auth_token": "tok_valid_demo",
            },
            "security": {
                "auth_failures_last_min": 0,
                "active_sessions": 1,
            },
            "attack_label": None,
        }


def main():
    parser = argparse.ArgumentParser(description="Orbital Sentinel AI — Day 1 satellite simulator")
    parser.add_argument("--hz", type=float, default=1.0, help="Frames per second (default 1)")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run (default: forever)")
    args = parser.parse_args()

    sim = SatelliteSimulator()
    interval = 1.0 / args.hz
    start = time.time()

    print(f"[orbital-sentinel] satellite_sim.py starting — {args.hz} Hz, NORMAL mode, satellite_id=sentinel-1")
    try:
        while args.duration is None or (time.time() - start) < args.duration:
            frame = sim.next_frame()
            print(json.dumps(frame))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[orbital-sentinel] stopped.")


if __name__ == "__main__":
    main()
