# Orbital Sentinel AI

Autonomous Satellite Cyber Defence, Communication Monitoring & Predictive Threat Intelligence System.
26-day hackathon build. See `docs/architecture.md` for full system design.

## Day 1 status
- [x] Repo scaffold
- [x] Telemetry frame schema frozen (`data/schemas/telemetry_frame.schema.json`)
- [x] NORMAL-mode simulator running (`simulator/satellite_sim.py`)
- [ ] Attack engine
- [ ] Redis Streams wiring
- [ ] ML pipeline
- [ ] Backend API
- [ ] Dashboard

## Quickstart
```bash
python simulator/satellite_sim.py --hz 2
```
