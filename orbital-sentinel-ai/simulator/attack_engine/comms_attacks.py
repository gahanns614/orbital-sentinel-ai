"""
Communication-layer attacks. These mutate the `comms` block of the frame.
"""

import random
from .base_attack import BaseAttack


class SignalJamming(BaseAttack):
    """
    Jamming: attacker floods the frequency with noise. Signature is a
    collapsing SNR and a rising noise floor, ramping in over ~5 seconds
    to look like a real jammer powering up rather than an instant switch.
    """
    name = "signal_jamming"
    category = "comms"
    severity = 0.8

    def apply(self, frame: dict, t_since_start: float) -> dict:
        ramp = min(1.0, t_since_start / 5.0)  # 0 -> 1 over first 5s
        frame["comms"]["signal_dbm"] -= 25 * ramp + random.uniform(0, 3)
        frame["comms"]["noise_floor_db"] += 20 * ramp + random.uniform(0, 2)
        frame["comms"]["snr_db"] = max(
            0.0, frame["comms"]["snr_db"] - 18 * ramp - random.uniform(0, 2)
        )
        frame["comms"]["packet_loss_pct"] = min(
            100.0, frame["comms"]["packet_loss_pct"] + 60 * ramp
        )
        return self.label_frame(frame)


class SignalSpoofing(BaseAttack):
    """
    Spoofing: a competing signal impersonates the legitimate downlink.
    Unlike jamming, the signal *looks* healthy (strength/SNR near-normal)
    but the auth token and checksum are inconsistent with a legitimate
    ground station -- this is the key detection hook (integrity check,
    not signal-quality check).
    """
    name = "signal_spoofing"
    category = "comms"
    severity = 0.9

    def apply(self, frame: dict, t_since_start: float) -> dict:
        # Signal quality stays plausible on purpose -- that's what makes
        # spoofing dangerous and different from jamming.
        frame["comms"]["signal_dbm"] += random.uniform(-2, 2)
        frame["command_stream"]["auth_token"] = "tok_spoofed_" + str(
            random.randint(1000, 9999)
        )
        frame["command_stream"]["checksum"] = f"{random.getrandbits(32):08x}"
        return self.label_frame(frame)


class CommLinkDegradation(BaseAttack):
    """
    Comm Link Degradation: a SLOW decay toward failure (equipment aging,
    antenna misalignment drifting, cumulative interference building up)
    -- distinct on purpose from SignalJamming's fast ~5s ramp. This is
    the attack the failure_predictor model is trained to catch EARLY,
    before the link fully fails, which only makes sense against a decay
    that unfolds over tens of seconds to minutes rather than an instant
    attack. Ramp duration defaults to 90s (vs jamming's 5s).
    """
    name = "comm_link_degradation"
    category = "comms"
    severity = 0.6

    def __init__(self, ramp_seconds: float = 90.0):
        self.ramp_seconds = ramp_seconds

    def apply(self, frame: dict, t_since_start: float) -> dict:
        ramp = min(1.0, t_since_start / self.ramp_seconds)
        frame["comms"]["signal_dbm"] -= 30 * ramp + random.uniform(0, 1.5)
        frame["comms"]["noise_floor_db"] += 15 * ramp + random.uniform(0, 1)
        frame["comms"]["snr_db"] = max(
            0.0, frame["comms"]["snr_db"] - 16 * ramp - random.uniform(0, 1)
        )
        frame["comms"]["latency_ms"] += 200 * ramp
        frame["comms"]["packet_loss_pct"] = min(
            100.0, frame["comms"]["packet_loss_pct"] + 50 * ramp
        )
        return self.label_frame(frame)
