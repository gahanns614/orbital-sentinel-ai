"""
ORBITAL SENTINEL AI — ws_bridge.py
Bridges Redis Streams -> WebSocket, so a browser (which can't talk to
Redis directly) can receive live telemetry frames as your simulator
publishes them.

Chain:
    scenario_runner.py / event_publisher.py -> Redis -> ws_bridge.py -> browser

Requires:
    pip install websockets redis

Run:
    python simulator/ws_bridge.py
Then, in ANOTHER terminal, run any attack scenario:
    python simulator/event_publisher.py --scenario full_assault --hz 2 --duration 30
Then open frontend/orbit-3d/index.html in your browser -- it will connect
automatically and react to whatever the simulator publishes, live.
"""

import asyncio
import json
import sys

try:
    import redis.asyncio as redis
except ImportError:
    print("ERROR: 'redis' package not installed. Run: pip install redis")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("ERROR: 'websockets' package not installed. Run: pip install websockets")
    sys.exit(1)

STREAM_NAME = "telemetry_frames"
WS_PORT = 8765

connected_clients = set()


async def redis_listener():
    """Continuously read new frames from Redis and fan them out to every
    connected browser tab."""
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    try:
        await r.ping()
    except Exception as e:
        print(f"ERROR: could not connect to Redis/Memurai on localhost:6379 -- {e}")
        sys.exit(1)

    print(f"[ws_bridge] connected to Redis. Watching stream '{STREAM_NAME}'.")
    last_id = "$"  # only new frames from now on

    while True:
        response = await r.xread({STREAM_NAME: last_id}, count=10, block=2000)
        if not response:
            continue
        for stream, messages in response:
            for msg_id, fields in messages:
                last_id = msg_id
                if connected_clients:
                    payload = fields["data"]  # already JSON-encoded frame
                    await asyncio.gather(
                        *(client.send(payload) for client in list(connected_clients)),
                        return_exceptions=True,
                    )


async def handle_client(websocket):
    connected_clients.add(websocket)
    print(f"[ws_bridge] browser connected. ({len(connected_clients)} total)")
    try:
        async for _ in websocket:
            pass  # this bridge is one-directional: Redis -> browser only
    finally:
        connected_clients.discard(websocket)
        print(f"[ws_bridge] browser disconnected. ({len(connected_clients)} total)")


async def main():
    print(f"[ws_bridge] starting WebSocket server on ws://localhost:{WS_PORT}")
    async with websockets.serve(handle_client, "localhost", WS_PORT):
        await redis_listener()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ws_bridge] stopped.")
