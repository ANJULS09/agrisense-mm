"""
Unit Tests for AgriSense-MM Services (Database, Router, Weather).
"""

import os
import pytest
from agrisense_mm.services.storage import AgriSenseDatabase
from agrisense_mm.services.router import InspectionRouter, compute_zone_risk_scores, haversine_distance_meters
from agrisense_mm.services.weather_api import WeatherService


def test_database_operations(tmp_path):
    db_file = str(tmp_path / "test_agrisense.db")
    db = AgriSenseDatabase(db_file)

    # Telemetry logging
    telem_id = db.log_telemetry({
        "vehicle_id": "test-rover",
        "latitude": 28.6140,
        "longitude": 77.2091,
        "speed_mps": 2.0,
        "battery_pct": 90.0,
    })
    assert telem_id > 0

    recent_telem = db.get_recent_telemetry(limit=5)
    assert len(recent_telem) == 1
    assert recent_telem[0]["vehicle_id"] == "test-rover"

    # Zone upsert
    db.upsert_zone_assessment({
        "zone_id": "Zone_A1",
        "grid_row": 0,
        "grid_col": 0,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "stress_class": "Water Stress",
        "severity": 65.0,
        "uncertainty": 0.25,
    })

    zones = db.get_all_zones()
    assert len(zones) == 1
    assert zones[0]["stress_class"] == "Water Stress"

    # Alerts
    al_id = db.log_alert({
        "zone_id": "Zone_A1",
        "severity": 65.0,
        "stress_class": "Water Stress",
        "message": "Critical water deficit",
    })
    assert al_id > 0
    alerts = db.get_recent_alerts(limit=5)
    assert len(alerts) == 1


def test_routing_engine():
    router = InspectionRouter()
    # Dummy field zones
    zones = [
        {"zone_id": "Zone_A1", "grid_row": 0, "grid_col": 0, "latitude": 28.6139, "longitude": 77.2090, "severity": 85.0, "uncertainty": 0.4},
        {"zone_id": "Zone_A2", "grid_row": 0, "grid_col": 1, "latitude": 28.6141, "longitude": 77.2092, "severity": 10.0, "uncertainty": 0.1},
        {"zone_id": "Zone_B1", "grid_row": 1, "grid_col": 0, "latitude": 28.6143, "longitude": 77.2090, "severity": 75.0, "uncertainty": 0.3},
        {"zone_id": "Zone_B2", "grid_row": 1, "grid_col": 1, "latitude": 28.6145, "longitude": 77.2092, "severity": 20.0, "uncertainty": 0.1},
    ]

    plan = router.plan_smart_inspection_route(
        current_lat=28.6138, current_lon=77.2089, zones=zones
    )

    assert plan["num_priority_zones"] > 0
    assert plan["smart_route_distance_m"] > 0.0
    assert len(plan["ordered_waypoints"]) > 0
    assert len(plan["route_polyline"]) > 1


def test_weather_service_fallback():
    service = WeatherService(timeout_sec=1)
    seq = service.fetch_14day_weather()
    assert seq.shape == (14, 7)

