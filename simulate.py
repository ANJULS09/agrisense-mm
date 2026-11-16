"""
AgriSense-MM: Standalone Sensor Telemetry Simulator.
Publishes live mobile sensing packets (GPS, IMU, weather, blur scores) to backend.
"""

import sys
from agrisense_mm.simulator.sensor_simulator import FieldVehicleSimulator

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sim = FieldVehicleSimulator(step_interval_sec=1.0)
    sim.run_simulation_loop(max_steps=steps)

