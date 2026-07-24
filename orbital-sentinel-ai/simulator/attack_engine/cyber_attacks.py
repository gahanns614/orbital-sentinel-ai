"""
Satellite cyber attacks operating on the command_stream block.
"""

import random
from .base_attack import BaseAttack


class ReplayAttack(BaseAttack):
    """
    Replay: attacker re-transmits a previously captured, legitimately
    authenticated command frame. Signature is a sequence number that goes
    BACKWARD or repeats instead of monotonically increasing -- classic
    replay detection is exactly this sequence-consistency check.
    """
    name = "replay_attack"
    category = "cyber"
    severity = 0.75

    def __init__(self):
        self._captured_seq = None

    def apply(self, frame: dict, t_since_start: float) -> dict:
        if self._captured_seq is None:
            # capture a "previously seen" seq the first time this attack fires
            self._captured_seq = frame["command_stream"]["seq"] - random.randint(50, 200)
        frame["command_stream"]["seq"] = self._captured_seq
        # timestamp stays "fresh" -- attacker replays the payload but sends
        # it now, which is what makes replay attacks subtle: everything
        # else about the frame looks legitimate.
        return self.label_frame(frame)
