"""
ORBITAL SENTINEL AI — backend/app/services/report_service.py
Day 16 scope: the "predictive threat intelligence" narrative layer.
Takes structured alert output (from risk_scoring.py) and calls the
Claude API to generate a human-readable ops report and mitigation
recommendation -- this turns
    {attack_type: "signal_jamming", risk_score: 82.2, confidence: 0.98}
into something a mission controller could actually read and act on.

This is deliberately a thin prompting layer, not a trained model --
per the blueprint, the LLM's job is narrative generation from ALREADY
-computed structured data, not detection itself. The detection is done
by the real ML models; this just explains it in English.

Requires:
    pip install anthropic

Set your API key as an environment variable before running the backend:
    setx ANTHROPIC_API_KEY "your-key-here"     (Windows, then restart terminal)
    export ANTHROPIC_API_KEY="your-key-here"   (Mac/Linux)
"""

import os
from typing import Optional

try:
    import anthropic
    _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    ANTHROPIC_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
except ImportError:
    ANTHROPIC_AVAILABLE = False
    _client = None

REPORT_MODEL = "claude-sonnet-4-6"


def _build_prompt(alert: dict) -> str:
    return f"""You are the AI threat-intelligence module for Orbital Sentinel AI, a satellite cyber defence system. A detection alert just fired. Write a SHORT mission-control style report (3-4 sentences max) for a human operator, followed by ONE concrete recommended mitigation action.

Alert data:
- Attack type: {alert['attack_type']}
- Risk level: {alert['risk_level']}
- Risk score: {alert['risk_score']}/100
- Classifier confidence: {alert['confidence']}
- Satellite: {alert.get('satellite_id', 'unknown')}

Write in the terse, factual tone of a real mission operations report -- no fluff, no exclamation points, no "I" statements. Format your response as exactly two parts separated by "---":
REPORT: <the 3-4 sentence situational report>
---
MITIGATION: <one concrete recommended action, one sentence>"""


def generate_report(alert: dict) -> dict:
    """Returns {report: str, mitigation: str, source: 'ai'|'fallback'}.
    Falls back to a deterministic templated report if the API isn't
    configured -- so the backend never breaks a demo just because a key
    is missing, it degrades gracefully instead."""
    if not ANTHROPIC_AVAILABLE:
        return _fallback_report(alert)

    try:
        response = _client.messages.create(
            model=REPORT_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": _build_prompt(alert)}],
        )
        text = response.content[0].text
        report_part, _, mitigation_part = text.partition("---")
        report = report_part.replace("REPORT:", "").strip()
        mitigation = mitigation_part.replace("MITIGATION:", "").strip()
        return {"report": report, "mitigation": mitigation, "source": "ai"}
    except Exception as e:
        # never let a report-generation failure take down the alert pipeline
        fallback = _fallback_report(alert)
        fallback["error"] = str(e)
        return fallback


def _fallback_report(alert: dict) -> dict:
    """Deterministic template used when the Anthropic API key isn't set
    or a call fails. This is what keeps the demo alive without network/
    API access -- worth keeping even after the real integration works."""
    templates = {
        "signal_jamming": (
            "Signal jamming detected on the downlink. Signal strength and SNR have "
            "degraded sharply, consistent with active RF interference.",
            "Switch to backup frequency band and increase transmit power if available.",
        ),
        "signal_spoofing": (
            "Command authentication failure detected. Signal quality remains nominal "
            "but authentication tokens do not match expected values, indicating a "
            "possible spoofed transmission.",
            "Reject all commands pending manual authentication verification from ground control.",
        ),
        "replay_attack": (
            "Command sequence anomaly detected. Incoming command sequence numbers are "
            "not advancing as expected, consistent with a replayed transmission.",
            "Enable strict sequence-number enforcement and discard the affected command batch.",
        ),
        "ddos": (
            "Ground station resource exhaustion detected. CPU and network load have "
            "spiked well above baseline while signal quality remains nominal.",
            "Enable traffic rate limiting on the ground station uplink receiver.",
        ),
        "brute_force": (
            "Repeated authentication failures detected on the command uplink, "
            "consistent with a credential brute-force attempt.",
            "Lock out the affected session and require multi-factor re-authentication.",
        ),
        "comm_link_degradation": (
            "Gradual communication link degradation detected. Signal and SNR trending "
            "downward over the recent observation window.",
            "Prepare backup communication path and consider preemptive safe-mode transition.",
        ),
    }
    report_text, mitigation_text = templates.get(
        alert["attack_type"],
        ("Unclassified anomaly detected requiring operator review.",
         "Flag for manual review by mission control."),
    )
    return {"report": report_text, "mitigation": mitigation_text, "source": "fallback"}
