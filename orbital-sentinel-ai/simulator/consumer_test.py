"""
ORBITAL SENTINEL AI — consumer_test.py
Day 3-4 scope: a minimal standalone consumer that proves the event bus
actually works -- it reads frames from Redis in a SEPARATE process from
the publisher, which is the whole point of decoupling via a stream.

This is a throwaway proof script, not the real backend ingestion service
(that comes later, in backend/app/services/). Run this in one terminal
while event_publisher.py runs in another.

Run:
    python simulator/consumer_test.py
"""

import json
import sys

try:
    import redis
except ImportError:
    print("ERROR: the 'redis' package isn't installed. Run: pip install redis")
    sys.exit(1)

STREAM_NAME = "telemetry_frames"


def main():
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    try:
        r.ping()
    except redis.exceptions.ConnectionError as e:
        print(f"ERROR: could not connect to Redis/Memurai -- {e}")
        sys.exit(1)

    print(f"[consumer_test] connected. Reading live from '{STREAM_NAME}'... (Ctrl+C to stop)")

    last_id = "$"  # "$" means: only new messages from now on
    try:
        while True:
            # blocks up to 5000ms waiting for new stream entries
            response = r.xread({STREAM_NAME: last_id}, count=10, block=5000)
            if not response:
                continue
            for stream, messages in response:
                for msg_id, fields in messages:
                    last_id = msg_id
                    frame = json.loads(fields["data"])
                    comms = frame["comms"]
                    print(
                        f"[{frame['timestamp']}] mode={frame['mode']:<8} "
                        f"label={str(frame['attack_label']):<16} "
                        f"signal={comms['signal_dbm']:.1f}dBm "
                        f"snr={comms['snr_db']:.1f}dB "
                        f"loss={comms['packet_loss_pct']:.1f}%"
                    )
    except KeyboardInterrupt:
        print("\n[consumer_test] stopped.")


if __name__ == "__main__":
    main()
