"""
ORBITAL SENTINEL AI — satellite_sim.py
Emits telemetry frames matching data/schemas/telemetry_frame.schema.json.

Orbit is now driven by real two-body Keplerian orbital mechanics for a
LEO (Low Earth Orbit) satellite -- not a random walk. Position is computed
from actual physics: semi-major axis, inclination, mean motion (Kepler's
third law), and Earth's rotation underneath the orbit. This produces a
real ground-track pattern (the characteristic westward-shifting sine wave
you see on satellite tracking sites), with latitude bounded exactly by
the orbital inclination and a correct ~95 minute period at 550km altitude.

DEMO TIME ACCELERATION: a real 550km LEO orbit takes ~95.5 minutes to
complete -- too slow to be watchable live. TIME_SCALE compresses simulated
orbital time relative to wall-clock time (default 60x => a full orbit
plays out in ~95 seconds). The physics itself is unmodified; only the
rate at which simulated time advances is scaled. Set TIME_SCALE = 1.0
for real-time-accurate orbital motion.

Run:
    python simulator/satellite_sim.py
    python simulator/satellite_sim.py --hz 5 --duration 30
"""

import argparse
import json
import math
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


class LEOOrbit:
    """
    Real two-body Keplerian propagator for a circular LEO orbit.
    No external dependencies (no sgp4/skyfield needed) -- this is the
    actual orbital mechanics math, self-contained.

    Circular-orbit assumption is standard for LEO comms/imaging satellites
    (eccentricity ~0 in practice), which keeps this exact rather than an
    approximation, while skipping full elliptical-orbit complexity that
    wouldn't add anything visible for this mission profile.
    """

    MU_EARTH = 398600.4418      # km^3/s^2, standard gravitational parameter
    R_EARTH = 6371.0            # km, mean Earth radius
    OMEGA_EARTH = 7.2921159e-5  # rad/s, sidereal Earth rotation rate

    def __init__(self, altitude_km: float = 550.0, inclination_deg: float = 53.0,
                 raan_deg: float = 0.0, mean_anomaly0_deg: float = 0.0,
                 time_scale: float = 60.0):
        self.altitude_km = altitude_km
        self.a = self.R_EARTH + altitude_km          # semi-major axis
        self.inc = math.radians(inclination_deg)
        self.raan = math.radians(raan_deg)
        self.M0 = math.radians(mean_anomaly0_deg)
        self.n = math.sqrt(self.MU_EARTH / self.a ** 3)  # mean motion, rad/s
        self.period_s = 2 * math.pi / self.n
        self.time_scale = time_scale

    def position_at(self, wall_clock_elapsed_s: float):
        """Return (lat_deg, lon_deg, alt_km) at the given elapsed wall-clock
        time, with TIME_SCALE applied to compress the orbit into a
        demo-friendly duration."""
        t = wall_clock_elapsed_s * self.time_scale

        M = self.M0 + self.n * t
        x_orb, y_orb = self.a * math.cos(M), self.a * math.sin(M)

        # rotate by inclination about the line of nodes
        x1 = x_orb
        y1 = y_orb * math.cos(self.inc)
        z1 = y_orb * math.sin(self.inc)

        # rotate by RAAN into the Earth-Centered Inertial (ECI) frame
        x_eci = x1 * math.cos(self.raan) - y1 * math.sin(self.raan)
        y_eci = x1 * math.sin(self.raan) + y1 * math.cos(self.raan)
        z_eci = z1

        # rotate into Earth-Centered Earth-Fixed (ECEF) to account for
        # Earth spinning underneath the orbit -- this is what produces the
        # realistic westward ground-track drift orbit-to-orbit.
        theta_g = self.OMEGA_EARTH * t
        x = x_eci * math.cos(theta_g) + y_eci * math.sin(theta_g)
        y = -x_eci * math.sin(theta_g) + y_eci * math.cos(theta_g)
        z = z_eci

        r = math.sqrt(x * x + y * y + z * z)
        lat = math.degrees(math.asin(z / r))
        lon = math.degrees(math.atan2(y, x))
        alt = r - self.R_EARTH
        return lat, lon, alt


class SatelliteSimulator:
    def __init__(self, satellite_id: str = "sentinel-1"):
        self.satellite_id = satellite_id
        self.seq = 10000
        self.start_time = time.time()

        # Real Keplerian LEO orbit: 550km altitude, 53deg inclination
        # (Starlink-like), ~95.5 min real orbital period, compressed 60x
        # for demo purposes (see TIME_SCALE note in module docstring).
        self.orbit = LEOOrbit(altitude_km=550.0, inclination_deg=53.0, time_scale=60.0)
        # small realistic altitude jitter (atmospheric drag / station-keeping
        # noise) layered on top of the exact physics -- kept tiny so it
        # doesn't distort the propagation.
        self.alt_jitter = MeanRevertingWalk(0, 0.05, floor=-1.5, ceiling=1.5)

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
        elapsed = time.time() - self.start_time
        lat, lon, alt = self.orbit.position_at(elapsed)
        alt += self.alt_jitter.step()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "satellite_id": self.satellite_id,
            "mode": "normal",
            "orbit": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "alt_km": round(alt, 2),
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
