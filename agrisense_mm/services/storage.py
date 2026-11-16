"""
AgriSense-MM: Storage Layer (SQLite / TimescaleDB compatible).
Maintains:
- Telemetry logs (GPS, IMU metrics, speed, battery, timestamp)
- Zone stress assessments (class, severity, uncertainty, forecast)
- Inspection route missions
- System alert dispatches
"""

import sqlite3
import json
import time
from typing import Dict, List, Optional, Any


class AgriSenseDatabase:
    def __init__(self, db_path: str = "agrisense_mm.db"):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Telemetry Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    vehicle_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    elevation REAL,
                    speed_mps REAL,
                    battery_pct REAL,
                    platform_state TEXT,
                    blur_quality_score REAL,
                    imu_variance REAL
                )
            """)

            # 2. Zone Assessment
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS zone_assessments (
                    zone_id TEXT PRIMARY KEY,
                    grid_row INTEGER,
                    grid_col INTEGER,
                    latitude REAL,
                    longitude REAL,
                    stress_class TEXT,
                    stress_class_idx INTEGER,
                    severity REAL,
                    uncertainty REAL,
                    forecast_3d TEXT,
                    forecast_7d TEXT,
                    risk_score REAL,
                    last_assessed_timestamp REAL,
                    recommended_action TEXT
                )
            """)

            # 3. System Alerts Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    zone_id TEXT,
                    severity REAL,
                    stress_class TEXT,
                    message TEXT,
                    dispatched_channel TEXT
                )
            """)

            conn.commit()

    def log_telemetry(self, data: Dict[str, Any]) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry (
                    timestamp, vehicle_id, latitude, longitude, elevation,
                    speed_mps, battery_pct, platform_state, blur_quality_score, imu_variance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.time()),
                data.get("vehicle_id", "rover-01"),
                data.get("latitude", 28.6139),
                data.get("longitude", 77.2090),
                data.get("elevation", 210.0),
                data.get("speed_mps", 1.5),
                data.get("battery_pct", 95.0),
                data.get("platform_state", "cruising"),
                data.get("blur_quality_score", 0.95),
                data.get("imu_variance", 0.12),
            ))
            conn.commit()
            return cursor.lastrowid

    def upsert_zone_assessment(self, zone_data: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO zone_assessments (
                    zone_id, grid_row, grid_col, latitude, longitude,
                    stress_class, stress_class_idx, severity, uncertainty,
                    forecast_3d, forecast_7d, risk_score, last_assessed_timestamp,
                    recommended_action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                zone_data["zone_id"],
                zone_data.get("grid_row", 0),
                zone_data.get("grid_col", 0),
                zone_data.get("latitude", 28.6139),
                zone_data.get("longitude", 77.2090),
                zone_data["stress_class"],
                zone_data.get("stress_class_idx", 0),
                zone_data["severity"],
                zone_data["uncertainty"],
                zone_data.get("forecast_3d", "Healthy"),
                zone_data.get("forecast_7d", "Healthy"),
                zone_data.get("risk_score", 0.0),
                zone_data.get("last_assessed_timestamp", time.time()),
                zone_data.get("recommended_action", "Nominal"),
            ))
            conn.commit()

    def get_all_zones(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM zone_assessments ORDER BY zone_id")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def log_alert(self, alert_data: Dict[str, Any]) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, zone_id, severity, stress_class, message, dispatched_channel)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert_data.get("timestamp", time.time()),
                alert_data["zone_id"],
                alert_data["severity"],
                alert_data["stress_class"],
                alert_data["message"],
                alert_data.get("dispatched_channel", "Telegram"),
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_telemetry(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

