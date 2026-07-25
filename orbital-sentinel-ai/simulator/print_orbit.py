"""
ORBITAL SENTINEL AI — print_orbit.py
Throwaway debug helper: reads JSON frames from stdin (piped from
scenario_runner.py or event_publisher.py) and prints just the orbit
fields, so you can visually confirm the Keplerian propagation is
producing smooth, realistic motion instead of random jumps.

Run:
    python simulator/scenario_runner.py --scenario kill_chain_demo --hz 2 --duration 10 | python simulator/print_orbit.py
"""

import sys
import json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        frame = json.loads(line)
    except json.JSONDecodeError:
        continue
    o = frame["orbit"]
    print(f"lat={o['lat']:8.3f}  lon={o['lon']:9.3f}  alt_km={o['alt_km']:7.2f}")
