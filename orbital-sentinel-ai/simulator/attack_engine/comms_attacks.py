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
