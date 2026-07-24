"""
Base interface every attack implements. Keeping this tiny and uniform is
what lets scenario_runner.py compose attacks (including simultaneous /
multi-stage ones) without knowing anything about each attack's internals.

Contract:
    - name: machine-readable id, used as the ground-truth attack_label
    - severity: 0.0-1.0, feeds the risk-scoring fusion logic later
    - apply(frame, t_since_start) -> mutated frame
        t_since_start lets attacks ramp up/down over time (e.g. jamming
        that gets progressively worse) instead of just flipping a switch.
"""

from abc import ABC, abstractmethod


class BaseAttack(ABC):
    name: str = "base_attack"
    category: str = "uncategorized"  # comms | cyber | network | security
    severity: float = 0.5            # 0.0 (nuisance) - 1.0 (critical)

    @abstractmethod
    def apply(self, frame: dict, t_since_start: float) -> dict:
        """Mutate and return the frame to reflect this attack's signature.
        t_since_start: seconds since this attack instance started (float).
        """
        raise NotImplementedError

    def label_frame(self, frame: dict) -> dict:
        frame["mode"] = "attack"
        frame["attack_label"] = self.name
        return frame
