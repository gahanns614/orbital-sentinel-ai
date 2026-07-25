"""
ORBITAL SENTINEL AI — backend/app/api/routes.py
Day 14-15 scope: REST endpoints matching the API design from the
original blueprint (Section 10). Auth/missions/scenario-trigger
endpoints are deliberately deferred (Day 17-18 per the plan) -- this is
the read-side API needed to get the dashboard talking to real inference
output instead of the raw ws_bridge relay.
"""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/telemetry/latest")
async def get_latest_telemetry(request: Request):
    inference = request.app.state.inference_service
    if inference.latest_enriched_frame is None:
        raise HTTPException(status_code=404, detail="No telemetry received yet")
    return inference.latest_enriched_frame


@router.get("/alerts")
async def get_alerts(request: Request, status: str = "active"):
    inference = request.app.state.inference_service
    if status == "active":
        return inference.get_active_alerts()
    return inference.get_all_alerts()


@router.get("/missions/{satellite_id}/status")
async def get_mission_status(satellite_id: str, request: Request):
    inference = request.app.state.inference_service
    frame = inference.latest_enriched_frame
    if frame is None or frame.get("satellite_id") != satellite_id:
        raise HTTPException(status_code=404, detail=f"No data for satellite '{satellite_id}'")
    return {
        "satellite_id": satellite_id,
        "mode": frame.get("mode"),
        "risk_score": frame["risk"]["risk_score"],
        "risk_level": frame["risk"]["risk_level"],
        "predicted_attack": frame["risk"]["predicted_attack"],
        "active_alerts": len(inference.get_active_alerts()),
    }
