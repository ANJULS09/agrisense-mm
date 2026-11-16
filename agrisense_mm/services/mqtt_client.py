"""
AgriSense-MM: MQTT Telemetry Broker Client.
Publishes and subscribes to vehicle telemetry, camera triggers, and emergency alerts.
Compatible with Mosquitto, EMQX, and HiveMQ brokers.
"""

import json
import time
from typing import Dict, Any, Callable, Optional
import paho.mqtt.client as mqtt


class AgriSenseMQTTClient:
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "agrisense_backend",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.connected = False

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected successfully to broker at {self.broker_host}:{self.broker_port}")
            # Subscribe to vehicle telemetry stream
            self.client.subscribe("agrisense/vehicle/telemetry")
        else:
            print(f"[MQTT] Connection failed with result code {rc}")

    def _on_disconnect(self, client, userdata, disconnect_flags, rc, properties=None):
        self.connected = False

    def connect(self) -> bool:
        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            # Non-blocking if Mosquitto broker is not currently running locally
            self.connected = False
            return False

    def publish_telemetry(self, telemetry_dict: Dict[str, Any]):
        if self.connected:
            payload = json.dumps(telemetry_dict)
            self.client.publish("agrisense/vehicle/telemetry", payload, qos=1)

    def publish_alert(self, alert_dict: Dict[str, Any]):
        if self.connected:
            payload = json.dumps(alert_dict)
            self.client.publish("agrisense/alerts/critical", payload, qos=2)

    def disconnect(self):
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()

