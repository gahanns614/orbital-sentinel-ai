"""
ORBITAL SENTINEL AI — backend/app/ws/connection_manager.py
Day 14-15 scope: manages WebSocket connections for the real backend.

This replaces simulator/ws_bridge.py's throwaway relay logic with a
proper, reusable connection manager -- same broadcast concept, but now
living inside the actual backend service rather than a standalone demo
script.
"""

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast_json(self, data: dict):
        """Send to every connected client. Dead connections are dropped
        silently rather than raising -- one disconnected dashboard tab
        shouldn't break the broadcast for everyone else."""
        dead = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                dead.add(connection)
        self.active_connections -= dead

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)
