"""
ORBITAL SENTINEL AI — backend/app/main.py
Day 14-15 scope: the real backend entrypoint. This is what
`simulator/ws_bridge.py` was always meant to be replaced by -- instead
of a bare Redis-to-WebSocket relay, this runs every frame through the
ML risk-scoring pipeline (Day 13) before broadcasting, and exposes a
real REST API alongside the WebSocket.

Chain now looks like:
    scenario_runner.py / event_publisher.py -> Redis -> THIS BACKEND
    (runs RiskScorer, manages alerts) -> WebSocket -> dashboard/3D view

Requires:
    pip install fastapi uvicorn redis

Run:
    uvicorn backend.app.main:app --reload --port 8000
Then point the 3D view / dashboard's WebSocket URL at:
    ws://localhost:8000/ws/live
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
from api.routes import router as api_router
from services.inference_service import InferenceService
from ws.connection_manager import ConnectionManager

try:
    import redis.asyncio as redis
except ImportError:
    print("ERROR: 'redis' package not installed. Run: pip install redis")
    sys.exit(1)


async def inference_loop(app: FastAPI):
    """Background task: continuously pull new frames from Redis, run
    them through the ML pipeline, and broadcast enriched results."""
    inference: InferenceService = app.state.inference_service
    manager: ConnectionManager = app.state.connection_manager

    while True:
        enriched_frames = await inference.process_new_frames()
        for frame in enriched_frames:
            await manager.broadcast_json(frame)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[backend] connecting to Redis...")
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    await redis_client.ping()
    print("[backend] connected. Loading ML models (this may take a few seconds)...")

    app.state.inference_service = InferenceService(redis_client)
    app.state.connection_manager = ConnectionManager()
    print("[backend] models loaded. Starting inference loop.")

    loop_task = asyncio.create_task(inference_loop(app))
    yield
    loop_task.cancel()
    await redis_client.aclose()


app = FastAPI(title="Orbital Sentinel AI Backend", lifespan=lifespan)

# permissive CORS for hackathon dev speed -- tighten before any real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    manager: ConnectionManager = websocket.app.state.connection_manager
    await manager.connect(websocket)
    print(f"[backend] dashboard connected. ({manager.connection_count} total)")
    try:
        while True:
            await websocket.receive_text()  # one-directional; just keeps the connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"[backend] dashboard disconnected. ({manager.connection_count} total)")
