"""
AgriSense-MM: FastAPI Backend Ingestion & Inference Service.
Endpoints:
- POST /api/telemetry: Live vehicle telemetry ingestion
- GET  /api/zones: Current multi-zone field state & stress risk map
- GET  /api/route: 2-Opt TSP inspection path from vehicle GPS
- POST /api/assess: Real-time multimodal stress assessment & alert trigger
- GET  /api/alerts: Alert dispatch feed (Telegram / webhook simulation)
- GET  /api/weather: Live 14-day Open-Meteo weather sequence
- WS   /ws/telemetry: Live WebSocket streaming of platform motion & GPS
"""

import os
import json
import time
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from agrisense_mm.services.storage import AgriSenseDatabase
from agrisense_mm.services.weather_api import WeatherService
from agrisense_mm.services.router import InspectionRouter, compute_zone_risk_scores
from agrisense_mm.data.constants import (
    DEFAULT_FARM_ORIGIN,
    FIELD_GRID_ROWS,
    FIELD_GRID_COLS,
    STRESS_CLASSES,
    RECOMMENDED_ACTIONS,
)

app = FastAPI(
    title="AgriSense-MM Backend Service",
    description="Multimodal Crop Stress Detection & Smart Transportation Route Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
db = AgriSenseDatabase()
weather_service = WeatherService()
router_engine = InspectionRouter()

# Connected WebSocket clients
active_websockets: List[WebSocket] = []

# Initialize default zone grid in database if empty
def _init_default_grid():
    existing = db.get_all_zones()
    if not existing:
        for r in range(FIELD_GRID_ROWS):
            for c in range(FIELD_GRID_COLS):
                zone_id = f"Zone_{chr(65 + r)}{c + 1}"
                lat = DEFAULT_FARM_ORIGIN[0] + r * 0.00025
                lon = DEFAULT_FARM_ORIGIN[1] + c * 0.00025
                db.upsert_zone_assessment({
                    "zone_id": zone_id,
                    "grid_row": r,
                    "grid_col": c,
                    "latitude": lat,
                    "longitude": lon,
                    "stress_class": "Healthy",
                    "stress_class_idx": 0,
                    "severity": 5.0,
                    "uncertainty": 0.08,
                    "forecast_3d": "Healthy",
                    "forecast_7d": "Healthy",
                    "risk_score": 0.05,
                    "last_assessed_timestamp": time.time(),
                    "recommended_action": RECOMMENDED_ACTIONS[0],
                })

_init_default_grid()


# Schemas
class TelemetryPayload(BaseModel):
    vehicle_id: str = "rover-01"
    latitude: float = 28.6139
    longitude: float = 77.2090
    elevation: float = 210.0
    speed_mps: float = 1.8
    battery_pct: float = 94.5
    platform_state: str = "cruising"
    blur_quality_score: float = 0.92
    imu_variance: float = 0.15


class AssessmentPayload(BaseModel):
    zone_id: str
    stress_class: str
    stress_class_idx: int = 0
    severity: float = Field(..., ge=0.0, le=100.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    forecast_3d: str = "Healthy"
    forecast_7d: str = "Healthy"


@app.get("/")
def root():
    return {
        "service": "AgriSense-MM Ingestion & Routing API",
        "status": "online",
        "authors": "Sohum Chandra Srivastava (562), Anjul Shukla (572)",
        "docs_url": "/docs",
    }


@app.post("/api/telemetry")
async def ingest_telemetry(payload: TelemetryPayload):
    data = payload.dict()
    data["timestamp"] = time.time()
    db.log_telemetry(data)

    # Broadcast to all live WebSocket listeners (e.g. Dashboard)
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

    return {"status": "success", "message": "Telemetry logged and broadcasted"}


@app.get("/api/zones")
def get_zones():
    zones = db.get_all_zones()
    scored_zones = compute_zone_risk_scores(zones)
    return {"count": len(scored_zones), "zones": scored_zones}


@app.get("/api/route")
def get_inspection_route(lat: Optional[float] = None, lon: Optional[float] = None):
    # Get current vehicle location from latest telemetry
    recent_telem = db.get_recent_telemetry(limit=1)
    if lat is None or lon is None:
        if recent_telem:
            lat = recent_telem[0]["latitude"]
            lon = recent_telem[0]["longitude"]
        else:
            lat, lon = DEFAULT_FARM_ORIGIN

    zones = db.get_all_zones()
    plan = router_engine.plan_smart_inspection_route(lat, lon, zones)
    return plan


@app.post("/api/assess")
def record_zone_assessment(payload: AssessmentPayload):
    data = payload.dict()
    data["last_assessed_timestamp"] = time.time()
    cls_idx = data.get("stress_class_idx", 0)
    data["recommended_action"] = RECOMMENDED_ACTIONS.get(cls_idx, "Monitor zone")

    # Save to DB
    db.upsert_zone_assessment(data)

    # Check alert threshold: Trigger alert if severity >= 65 or Water/Disease/Pest stress
    if data["severity"] >= 65.0 or (data["stress_class"] != "Healthy" and data["severity"] >= 40.0):
        alert_msg = (
            f"ALERT [{data['zone_id']}]: High {data['stress_class']} detected "
            f"(Severity: {data['severity']:.1f}/100, Uncertainty: {data['uncertainty']:.2f}). "
            f"Action: {data['recommended_action']}"
        )
        db.log_alert({
            "zone_id": data["zone_id"],
            "severity": data["severity"],
            "stress_class": data["stress_class"],
            "message": alert_msg,
            "dispatched_channel": "Telegram Bot & Dashboard Pill",
        })

    return {"status": "success", "zone_id": data["zone_id"]}


@app.get("/api/alerts")
def get_alerts(limit: int = 15):
    alerts = db.get_recent_alerts(limit=limit)
    return {"count": len(alerts), "alerts": alerts}


@app.get("/api/weather")
def get_live_weather(lat: float = DEFAULT_FARM_ORIGIN[0], lon: float = DEFAULT_FARM_ORIGIN[1]):
    weather_seq = weather_service.fetch_14day_weather(lat, lon)
    return {
        "latitude": lat,
        "longitude": lon,
        "days": 14,
        "features": [
            "temperature_c", "humidity_pct", "soil_moisture_pct",
            "rainfall_mm", "solar_radiation_wm2", "wind_speed_ms", "ndvi"
        ],
        "sequence": weather_seq.tolist(),
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            # Keepalive listener
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

