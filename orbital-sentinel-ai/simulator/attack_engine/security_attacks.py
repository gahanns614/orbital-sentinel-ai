"""
Security/access-control attacks operating on the `security` block.
"""

import random
from .base_attack import BaseAttack


class BruteForceAttack(BaseAttack):
    """
    Brute force: rapid repeated authentication attempts against the
    ground station uplink. Signature is auth_failures_last_min spiking
    far above baseline (normally 0), with occasional active_sessions
    jitter as attempts briefly "succeed" against weak/reused credentials.
    """
    name = "brute_force"
    category = "security"
    severity = 0.6

    def apply(self, frame: dict, t_since_start: float) -> dict:
        frame["security"]["auth_failures_last_min"] = random.randint(15, 40)
        if random.random() < 0.1:
            frame["security"]["active_sessions"] += 1
        return self.label_frame(frame)
