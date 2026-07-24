from .comms_attacks import SignalJamming, SignalSpoofing
from .cyber_attacks import ReplayAttack
from .network_attacks import DDoSAttack
from .security_attacks import BruteForceAttack

# Central registry -- scenario_runner.py and the Day 9 data-generation
# script both key off this instead of hardcoding imports. Add each new
# attack here as it's implemented.
ATTACK_REGISTRY = {
    SignalJamming.name: SignalJamming,
    SignalSpoofing.name: SignalSpoofing,
    ReplayAttack.name: ReplayAttack,
    DDoSAttack.name: DDoSAttack,
    BruteForceAttack.name: BruteForceAttack,
}

__all__ = ["ATTACK_REGISTRY", "SignalJamming", "SignalSpoofing", "ReplayAttack",
           "DDoSAttack", "BruteForceAttack"]
