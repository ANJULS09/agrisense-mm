"""
AgriSense-MM: Real-Time Mobile Sensing Simulator.
Simulates a rover / drone navigating through agricultural zones,
transmitting time-synchronized GPS, IMU, Weather, and Camera streams
over HTTP REST and MQTT.
"""

import os
import time
import math
import random
import requests
import numpy as np
from typing import Dict, List, Optional, Any

from agrisense_mm.data.constants import (
    DEFAULT_FARM_ORIGIN,
    FIELD_GRID_ROWS,
    FIELD_GRID_COLS,
    PLATFORM_STATES,
)
from agrisense_mm.data.generator import (
    generate_imu_window,
    generate_environmental_sequence,
    generate_synthetic_leaf_image,
)


class FieldVehicleSimulator:
    def __init__(
        self,
        api_base_url: str = "http://127.0.0.1:8000",
        vehicle_id: str = "rover-alpha-01",
        origin_lat: float = DEFAULT_FARM_ORIGIN[0],
        origin_lon: float = DEFAULT_FARM_ORIGIN[1],
        step_interval_sec: float = 1.0,
    ):
        self.api_base_url = api_base_url
        self.vehicle_id = vehicle_id
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.step_interval_sec = step_interval_sec

        # State tracking
        self.current_lat = origin_lat
        self.current_lon = origin_lon
        self.battery_pct = 98.0
        self.heading_deg = 0.0
        self.speed_mps = 1.8
        self.step_count = 0

        # Build trajectory waypoints through the field grid
        self.waypoints = []
        for r in range(FIELD_GRID_ROWS):
            cols_range = range(FIELD_GRID_COLS) if r % 2 == 0 else range(FIELD_GRID_COLS - 1, -1, -1)
            for c in cols_range:
                w_lat = origin_lat + r * 0.00025
                w_lon = origin_lon + c * 0.00025
                self.waypoints.append((w_lat, w_lon, f"Zone_{chr(65 + r)}{c + 1}", r, c))

        self.current_wp_idx = 0

    def step(self) -> Dict[str, Any]:
        """Advances vehicle motion by one simulation step and generates telemetry packet."""
        target_lat, target_lon, zone_name, r, c = self.waypoints[self.current_wp_idx]

        # Interpolate movement towards waypoint
        d_lat = target_lat - self.current_lat
        d_lon = target_lon - self.current_lon
        dist = math.hypot(d_lat, d_lon)

        # Platform state determination
        if dist < 0.00005:  # Arrived at waypoint
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            platform_state = "turning"
            speed = 0.8
        else:
            # Move towards waypoint
            step_fraction = min(0.35, 0.00003 / (dist + 1e-9))
            self.current_lat += d_lat * step_fraction
            self.current_lon += d_lon * step_fraction
            self.heading_deg = (math.degrees(math.atan2(d_lon, d_lat)) + 360.0) % 360.0

            # Occasionally simulate high vibration on rough terrain
            if random.random() < 0.12:
                platform_state = "high_vibration"
                speed = 1.2
            else:
                platform_state = "cruising"
                speed = 1.8

        self.speed_mps = speed
        self.battery_pct = max(10.0, self.battery_pct - 0.02)
        self.step_count += 1

        # Generate realistic IMU waveform
        imu_wave = generate_imu_window(platform_state)
        imu_variance = float(np.var(imu_wave[:3, :]))

        # Image blur quality score (derived from IMU vibration level)
        if platform_state == "high_vibration":
            blur_q = float(np.random.uniform(0.18, 0.45))
        elif platform_state == "turning":
            blur_q = float(np.random.uniform(0.60, 0.80))
        else:
            blur_q = float(np.random.uniform(0.88, 0.99))

        packet = {
            "timestamp": time.time(),
            "vehicle_id": self.vehicle_id,
            "latitude": round(self.current_lat, 6),
            "longitude": round(self.current_lon, 6),
            "elevation": 210.0 + random.uniform(-0.3, 0.3),
            "speed_mps": round(self.speed_mps, 2),
            "heading_deg": round(self.heading_deg, 1),
            "battery_pct": round(self.battery_pct, 1),
            "platform_state": platform_state,
            "blur_quality_score": round(blur_q, 4),
            "imu_variance": round(imu_variance, 4),
            "current_zone": zone_name,
            "step": self.step_count,
        }

        return packet

    def transmit_telemetry(self, packet: Dict[str, Any]) -> bool:
        """Sends telemetry packet to backend API."""
        try:
            url = f"{self.api_base_url}/api/telemetry"
            resp = requests.post(url, json=packet, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def run_simulation_loop(self, max_steps: int = 50):
        """Runs the continuous simulation loop."""
        print(f"Starting Field Vehicle Telemetry Simulator ({self.vehicle_id})...")
        for i in range(max_steps):
            packet = self.step()
            success = self.transmit_telemetry(packet)
            status_symbol = "[OK]" if success else "[LOCAL]"
            print(
                f"{status_symbol} Step {i+1:02d} | Zone: {packet['current_zone']} | "
                f"GPS: ({packet['latitude']:.5f}, {packet['longitude']:.5f}) | "
                f"State: {packet['platform_state']:<14} | Blur Q: {packet['blur_quality_score']:.3f} | Batt: {packet['battery_pct']}%"
            )
            time.sleep(self.step_interval_sec)


if __name__ == "__main__":
    sim = FieldVehicleSimulator()
    sim.run_simulation_loop(max_steps=10)

