"""
Network-layer attacks. DDoS is modeled as ground-station-side resource
exhaustion: the satellite's uplink receiver gets flooded with junk
requests, which shows up as elevated CPU load and rising latency/packet
loss on the comms link, distinct from jamming because signal strength
itself stays normal (it's a compute/traffic problem, not an RF problem).
"""

import random
from .base_attack import BaseAttack


class DDoSAttack(BaseAttack):
    name = "ddos"
    category = "network"
    severity = 0.7

    def apply(self, frame: dict, t_since_start: float) -> dict:
        ramp = min(1.0, t_since_start / 8.0)
        frame["compute"]["cpu_pct"] = min(
            100.0, frame["compute"]["cpu_pct"] + 55 * ramp + random.uniform(0, 5)
        )
        frame["compute"]["mem_pct"] = min(
            100.0, frame["compute"]["mem_pct"] + 30 * ramp
        )
        frame["comms"]["latency_ms"] += 400 * ramp + random.uniform(0, 50)
        frame["comms"]["packet_loss_pct"] = min(
            100.0, frame["comms"]["packet_loss_pct"] + 35 * ramp
        )
        # signal_dbm/snr_db deliberately left untouched -- this is what
        # distinguishes DDoS from jamming in the feature space.
        return self.label_frame(frame)
