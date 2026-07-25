"""
ORBITAL SENTINEL AI — backend/app/services/inference_service.py
Day 14-15 scope: the real backend's core loop. Replaces the demo
ws_bridge.py's pure pass-through relay with actual ML inference:
consumes raw telemetry from Redis, runs it through RiskScorer (Day 13),
and produces ENRICHED frames (raw telemetry + risk_score + predicted
attack + alert status) for the dashboard/3D view to consume.

Alert dedup logic: a HIGH/CRITICAL risk frame only creates a NEW alert
if the predicted attack differs from the currently-open alert, or if
there's no currently-open alert. This avoids spamming one alert per
frame during a sustained attack (e.g. 40 frames of ongoing jamming
should be ONE alert, not 40).
"""

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml"))
from risk_scoring import RiskScorer, risk_level_for_score  # noqa: E402

try:
    from .report_service import generate_report
except ImportError:
    # fallback for running this file standalone/from a test harness
    # rather than as part of the `services` package
    sys.path.insert(0, str(Path(__file__).parent))
    from report_service import generate_report

STREAM_NAME = "telemetry_frames"
ALERT_TRIGGER_LEVELS = {"HIGH", "CRITICAL"}
MAX_ALERT_HISTORY = 200


class InferenceService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.scorer = RiskScorer()
        self.latest_enriched_frame: Optional[dict] = None
        self.alerts: list[dict] = []
        self._open_alert: Optional[dict] = None
        self._last_stream_id = "$"  # only new frames from service startup onward

    def _enrich(self, frame: dict) -> dict:
        result = self.scorer.score(frame)
        enriched = dict(frame)  # shallow copy, don't mutate the raw frame
        enriched["risk"] = result
        return enriched

    def _update_alerts(self, enriched_frame: dict):
        risk = enriched_frame["risk"]
        level = risk["risk_level"]
        predicted = risk["predicted_attack"]

        if level not in ALERT_TRIGGER_LEVELS:
            # risk dropped back down -- close any open alert
            if self._open_alert is not None:
                self._open_alert["status"] = "resolved"
                self._open_alert["resolved_at"] = enriched_frame["timestamp"]
                self._open_alert = None
            return

        # risk is HIGH/CRITICAL: open a new alert only if this is a
        # different attack than whatever's currently open (dedup)
        if self._open_alert is None or self._open_alert["attack_type"] != predicted:
            if self._open_alert is not None:
                self._open_alert["status"] = "superseded"
            alert = {
                "id": str(uuid.uuid4())[:8],
                "attack_type": predicted,
                "risk_level": level,
                "risk_score": risk["risk_score"],
                "confidence": risk["classifier_confidence"],
                "opened_at": enriched_frame["timestamp"],
                "status": "active",
                "satellite_id": enriched_frame.get("satellite_id"),
            }
            # AI report generation happens once per NEW alert, not per
            # frame -- keeps API calls infrequent (only on state
            # transitions) rather than on every single scored frame.
            # NOTE: this is a synchronous/blocking call for simplicity;
            # if it visibly stalls the inference loop during a demo,
            # move it to a background task -- flagging honestly rather
            # than over-engineering async here under time pressure.
            report_result = generate_report(alert)
            alert["ai_report"] = report_result["report"]
            alert["ai_mitigation"] = report_result["mitigation"]
            alert["report_source"] = report_result["source"]

            self.alerts.append(alert)
            self.alerts = self.alerts[-MAX_ALERT_HISTORY:]
            self._open_alert = alert
        else:
            # same ongoing attack -- just keep the open alert's numbers current
            self._open_alert["risk_score"] = risk["risk_score"]
            self._open_alert["confidence"] = risk["classifier_confidence"]

    async def process_new_frames(self):
        """Read any new frames from Redis since last call, score them,
        update alerts, and return the list of enriched frames (for the
        caller to broadcast over WebSocket)."""
        import json

        response = await self.redis.xread(
            {STREAM_NAME: self._last_stream_id}, count=50, block=1000
        )
        if not response:
            return []

        enriched_frames = []
        for stream, messages in response:
            for msg_id, fields in messages:
                self._last_stream_id = msg_id
                frame = json.loads(fields["data"])
                enriched = self._enrich(frame)
                self._update_alerts(enriched)
                self.latest_enriched_frame = enriched
                enriched_frames.append(enriched)

        return enriched_frames

    def get_active_alerts(self) -> list[dict]:
        return [a for a in self.alerts if a["status"] == "active"]

    def get_all_alerts(self) -> list[dict]:
        return self.alerts
