"""
AgriSense-MM: Multimodal Paired Dataset Generator.
Synthesizes aligned multimodal data (leaf images, 14-day weather sequences,
IMU motion profiles, GPS/soil data, multi-task stress & forecasting labels).
Also computes Laplacian variance for motion blur quality supervision.
"""

import os
import json
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Dict, List, Optional, Tuple, Any

from agrisense_mm.data.constants import (
    STRESS_CLASSES,
    NUM_CLASSES,
    PLATFORM_STATES,
    NUM_PLATFORM_STATES,
    ENV_WINDOW_DAYS,
    NUM_ENV_FEATURES,
    IMU_SAMPLE_POINTS,
    NUM_IMU_CHANNELS,
    DEFAULT_FARM_ORIGIN,
    FIELD_GRID_ROWS,
    FIELD_GRID_COLS,
    TOTAL_ZONES,
)


def compute_laplacian_blur_score(image_rgb: np.ndarray) -> float:
    """
    Computes normalized blur quality score q in [0, 1] using Laplacian variance.
    Higher variance -> sharp image (q -> 1.0).
    Lower variance -> blurry/vibrated image (q -> 0.0).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize with empirical upper bound (variance >= 400 is considered razor sharp)
    quality = float(np.clip(variance / 400.0, 0.05, 1.0))
    return quality


def generate_synthetic_leaf_image(
    stress_idx: int,
    severity: float,
    platform_state: str,
    img_size: int = 224,
) -> Tuple[np.ndarray, float]:
    """
    Synthesizes a realistic leaf image reflecting the crop stress class,
    and injects motion blur / camera shake if platform_state has high vibration.

    Returns:
        (image_rgb as uint8 numpy array [H, W, 3], normalized quality score q in [0, 1])
    """
    # Base background (neutral dark soil or field background)
    img = Image.new("RGB", (img_size, img_size), color=(38, 28, 20))
    draw = ImageDraw.Draw(img)

    cx, cy = img_size // 2, img_size // 2
    rx, ry = int(img_size * 0.38), int(img_size * 0.44)

    # Base leaf color (nominal healthy green)
    base_leaf_color = (46, 139, 87)  # Sea Green
    if stress_idx == 1:  # Water stress (wilting, pale grayish-green / yellowing tip)
        base_leaf_color = (95, 125, 75)
    elif stress_idx == 2:  # Nutrient deficiency (chlorosis: pale lime green to yellow)
        base_leaf_color = (145, 165, 60)
    elif stress_idx == 3:  # Heat stress (dry olive green with scorched edges)
        base_leaf_color = (120, 130, 50)
    elif stress_idx == 4:  # Disease stress (dull mottled olive)
        base_leaf_color = (60, 100, 60)
    elif stress_idx == 5:  # Pest damage (chewed foliage)
        base_leaf_color = (50, 120, 70)

    # Draw primary leaf ellipse
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=base_leaf_color, outline=(25, 70, 30), width=2)

    # Central midrib vein
    draw.line([(cx, cy - ry + 10), (cx, cy + ry - 10)], fill=(70, 180, 80), width=3)
    # Lateral secondary veins
    for dy in range(-ry + 30, ry - 30, 25):
        y_pos = cy + dy
        dx_val = int(rx * 0.7 * (1.0 - (abs(dy) / ry) ** 2))
        draw.line([(cx, y_pos), (cx - dx_val, y_pos - 15)], fill=(65, 165, 75), width=2)
        draw.line([(cx, y_pos), (cx + dx_val, y_pos - 15)], fill=(65, 165, 75), width=2)

    # Add pathology signatures based on stress class and severity
    np_img = np.array(img).astype(np.float32)

    if stress_idx == 1:
        # Water Stress: desiccated brownish/yellow wilting margins
        wilting_intensity = (severity / 100.0)
        margin_mask = np.zeros((img_size, img_size), dtype=np.float32)
        cv2.ellipse(margin_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        inner_mask = np.zeros((img_size, img_size), dtype=np.float32)
        cv2.ellipse(inner_mask, (cx, cy), (int(rx * 0.75), int(ry * 0.75)), 0, 0, 360, 1.0, -1)
        edge_zone = (margin_mask - inner_mask) * wilting_intensity

        np_img[:, :, 0] += edge_zone * 80.0   # add reddish-brown
        np_img[:, :, 1] -= edge_zone * 30.0   # drop lush green
        np_img[:, :, 2] -= edge_zone * 20.0

    elif stress_idx == 2:
        # Nutrient Deficiency: Interveinal chlorosis (yellowing between veins)
        yellow_tint = (severity / 100.0) * 90.0
        # Add high-frequency yellow chlorosis
        noise = np.random.uniform(0.6, 1.0, (img_size, img_size))
        leaf_mask = np.zeros((img_size, img_size), dtype=np.float32)
        cv2.ellipse(leaf_mask, (cx, cy), (int(rx * 0.9), int(ry * 0.9)), 0, 0, 360, 1.0, -1)
        np_img[:, :, 0] += leaf_mask * noise * yellow_tint * 0.8
        np_img[:, :, 1] += leaf_mask * noise * yellow_tint * 0.7
        np_img[:, :, 2] -= leaf_mask * noise * yellow_tint * 0.4

    elif stress_idx == 3:
        # Heat Stress: Scorched brown patches & desiccation
        num_scorch = int(3 + 5 * (severity / 100.0))
        for _ in range(num_scorch):
            px = int(cx + np.random.uniform(-rx * 0.6, rx * 0.6))
            py = int(cy + np.random.uniform(-ry * 0.6, ry * 0.6))
            radius = int(np.random.uniform(10, 22))
            scorch_mask = np.zeros((img_size, img_size), dtype=np.float32)
            cv2.circle(scorch_mask, (px, py), radius, 1.0, -1)
            scorch_mask = cv2.GaussianBlur(scorch_mask, (15, 15), 0)
            np_img[:, :, 0] = np_img[:, :, 0] * (1 - scorch_mask) + scorch_mask * 139  # SaddleBrown
            np_img[:, :, 1] = np_img[:, :, 1] * (1 - scorch_mask) + scorch_mask * 69
            np_img[:, :, 2] = np_img[:, :, 2] * (1 - scorch_mask) + scorch_mask * 19

    elif stress_idx == 4:
        # Disease Stress: Concentric fungal leaf spots with chlorotic yellow halos
        num_lesions = int(4 + 8 * (severity / 100.0))
        for _ in range(num_lesions):
            px = int(cx + np.random.uniform(-rx * 0.65, rx * 0.65))
            py = int(cy + np.random.uniform(-ry * 0.65, ry * 0.65))
            halo_r = int(np.random.uniform(12, 20))
            center_r = int(halo_r * 0.5)

            # Yellow Halo
            cv2.circle(np_img, (px, py), halo_r, (180, 190, 40), -1)
            # Necrotic Dark Brown Center
            cv2.circle(np_img, (px, py), center_r, (50, 30, 20), -1)
            # Spore center
            cv2.circle(np_img, (px, py), max(1, center_r // 2), (20, 10, 10), -1)

    elif stress_idx == 5:
        # Pest Damage: Chewed irregular holes and serrated leaf perforations
        num_bites = int(3 + 6 * (severity / 100.0))
        for _ in range(num_bites):
            px = int(cx + np.random.uniform(-rx * 0.6, rx * 0.6))
            py = int(cy + np.random.uniform(-ry * 0.6, ry * 0.6))
            r_bite = int(np.random.uniform(8, 18))
            # Hole exposes soil background
            cv2.circle(np_img, (px, py), r_bite, (38, 28, 20), -1)
            # Chewed necrotic border
            cv2.circle(np_img, (px, py), r_bite + 2, (70, 50, 30), 1)

    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    # Apply vibration / motion blur if vehicle platform is experiencing turbulence
    if platform_state == "high_vibration":
        # Directional motion blur kernel
        kernel_size = random.choice([9, 13, 17])
        angle = random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        for k in range(kernel_size):
            offset = k - kernel_size // 2
            x = int(round(kernel_size // 2 + offset * cos_a))
            y = int(round(kernel_size // 2 + offset * sin_a))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        kernel /= (kernel.sum() + 1e-6)
        np_img = cv2.filter2D(np_img, -1, kernel)
    elif platform_state == "turning":
        # Slight centrifugal blur
        np_img = cv2.GaussianBlur(np_img, (5, 5), 1.2)

    # Compute actual Laplacian blur quality score q
    q_score = compute_laplacian_blur_score(np_img)

    return np_img, q_score


def generate_environmental_sequence(
    stress_idx: int,
    severity: float,
    days: int = ENV_WINDOW_DAYS,
) -> np.ndarray:
    """
    Generates realistic 14-day environmental time-series [14, 7]
    Features: [temp_c, humidity_pct, soil_moisture_pct, rainfall_mm, solar_rad, wind_speed, ndvi]
    """
    timeline = np.zeros((days, NUM_ENV_FEATURES), dtype=np.float32)

    # Base nominal climatological values
    temp_base = 25.0
    hum_base = 65.0
    soil_base = 38.0
    rain_base = 2.0
    solar_base = 550.0
    wind_base = 3.5
    ndvi_base = 0.82

    for d in range(days):
        day_prog = d / max(1, days - 1)  # 0.0 -> 1.0

        if stress_idx == 0:  # Healthy
            t = temp_base + np.random.normal(0, 1.5)
            h = hum_base + np.random.normal(0, 4.0)
            s = soil_base + np.random.normal(0, 2.0)
            r = max(0.0, np.random.exponential(1.5) if d % 4 == 0 else 0.0)
            sol = solar_base + np.random.normal(0, 30.0)
            w = wind_base + np.random.normal(0, 1.0)
            ndvi = ndvi_base + np.random.normal(0, 0.02)

        elif stress_idx == 1:  # Water stress (drought / extreme drydown)
            t = temp_base + 4.0 + 3.0 * day_prog + np.random.normal(0, 1.5)
            h = max(20.0, hum_base - 25.0 - 15.0 * day_prog + np.random.normal(0, 3.0))
            # Soil moisture depleting sharply
            s = max(8.0, soil_base - (25.0 * (severity / 100.0) * day_prog) + np.random.normal(0, 1.5))
            r = 0.0  # Prolonged dry spell
            sol = solar_base + 120.0 + np.random.normal(0, 30.0)
            w = wind_base + 1.5 + np.random.normal(0, 1.0)
            ndvi = max(0.25, ndvi_base - (0.35 * (severity / 100.0) * day_prog) + np.random.normal(0, 0.02))

        elif stress_idx == 2:  # Nutrient deficiency (N/P/K depletion)
            t = temp_base + np.random.normal(0, 1.5)
            h = hum_base + np.random.normal(0, 4.0)
            s = soil_base + np.random.normal(0, 2.0)  # soil water is adequate
            r = max(0.0, np.random.exponential(2.0) if d % 3 == 0 else 0.0)
            sol = solar_base + np.random.normal(0, 25.0)
            w = wind_base + np.random.normal(0, 0.8)
            # NDVI steadily declines despite water
            ndvi = max(0.35, ndvi_base - (0.28 * (severity / 100.0) * day_prog) + np.random.normal(0, 0.015))

        elif stress_idx == 3:  # Heat stress (extreme heatwave)
            t = temp_base + 12.0 + 5.0 * day_prog + np.random.normal(0, 1.5)  # 37 - 44 deg C
            h = max(18.0, hum_base - 35.0 + np.random.normal(0, 2.5))
            s = max(12.0, soil_base - 18.0 * day_prog + np.random.normal(0, 2.0))
            r = 0.0
            sol = solar_base + 280.0 + np.random.normal(0, 35.0)  # > 800 W/m^2
            w = wind_base + 2.5 + np.random.normal(0, 1.2)
            ndvi = max(0.30, ndvi_base - (0.30 * (severity / 100.0) * day_prog) + np.random.normal(0, 0.02))

        elif stress_idx == 4:  # Disease stress (prolonged humidity & rain favors fungal proliferation)
            t = 23.0 + np.random.normal(0, 1.0)  # Moderate incubation temp (22-25 C)
            h = min(98.0, 85.0 + 10.0 * day_prog + np.random.normal(0, 2.0))  # Persistent high humidity
            s = min(60.0, soil_base + 12.0 + np.random.normal(0, 2.0))
            r = max(0.0, np.random.exponential(12.0) if d % 2 == 0 else 1.5)  # Frequent rainfall
            sol = solar_base - 180.0 + np.random.normal(0, 40.0)  # Cloudy
            w = wind_base - 1.0 + np.random.normal(0, 0.6)
            ndvi = max(0.30, ndvi_base - (0.32 * (severity / 100.0) * (day_prog ** 1.5)) + np.random.normal(0, 0.02))

        elif stress_idx == 5:  # Pest damage (sudden defoliation event)
            t = temp_base + np.random.normal(0, 1.5)
            h = hum_base + np.random.normal(0, 3.5)
            s = soil_base + np.random.normal(0, 2.0)
            r = max(0.0, np.random.exponential(2.0) if d % 4 == 0 else 0.0)
            sol = solar_base + np.random.normal(0, 25.0)
            w = wind_base + np.random.normal(0, 0.9)
            # Sharp cliff in NDVI in the final 4 days as pests chew foliage
            pest_defoliation = 0.0 if d < (days - 4) else ((d - (days - 4)) / 4.0) * (severity / 100.0) * 0.40
            ndvi = max(0.25, ndvi_base - pest_defoliation + np.random.normal(0, 0.015))

        timeline[d] = [t, h, s, r, sol, w, ndvi]

    return timeline


def generate_imu_window(
    platform_state: str,
    points: int = IMU_SAMPLE_POINTS,
) -> np.ndarray:
    """
    Generates 6-channel IMU time series (100 Hz for 1 second = [6, 100]).
    Channels: [ax, ay, az, gx, gy, gz].
    """
    t = np.linspace(0, 1.0, points)
    imu = np.zeros((NUM_IMU_CHANNELS, points), dtype=np.float32)

    # Base gravity on Z axis (vehicle level: az ~ 9.81 m/s^2)
    imu[2, :] = 9.81

    if platform_state == "cruising":
        # Smooth forward travel: small periodic vibration from motor / rover tracks
        freq = 12.0  # Hz
        imu[0, :] = 0.2 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.08, points)
        imu[1, :] = 0.15 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.08, points)
        imu[2, :] += 0.3 * np.sin(2 * np.pi * 2 * freq * t) + np.random.normal(0, 0.1, points)
        # Gyroscope
        imu[3, :] = np.random.normal(0, 0.02, points)
        imu[4, :] = np.random.normal(0, 0.02, points)
        imu[5, :] = np.random.normal(0, 0.03, points)

    elif platform_state == "turning":
        # Lateral centripetal acceleration on Y, yaw rate on Gyro Z
        turn_accel = 1.8
        imu[0, :] = np.random.normal(0, 0.12, points)
        imu[1, :] = turn_accel + np.random.normal(0, 0.15, points)
        imu[2, :] += np.random.normal(0, 0.15, points)
        # Gyro Z has strong rotational velocity (rad/s)
        imu[3, :] = np.random.normal(0, 0.05, points)
        imu[4, :] = np.random.normal(0, 0.05, points)
        imu[5, :] = 0.85 + np.random.normal(0, 0.08, points)

    elif platform_state == "hovering":
        # Stationary drone hovering: high-frequency motor blade jitter, zero net acceleration
        high_freq = 60.0  # Hz
        imu[0, :] = 0.15 * np.sin(2 * np.pi * high_freq * t) + np.random.normal(0, 0.04, points)
        imu[1, :] = 0.15 * np.cos(2 * np.pi * high_freq * t) + np.random.normal(0, 0.04, points)
        imu[2, :] += 0.25 * np.sin(2 * np.pi * high_freq * t) + np.random.normal(0, 0.05, points)
        imu[3, :] = 0.05 * np.sin(2 * np.pi * high_freq * t) + np.random.normal(0, 0.01, points)
        imu[4, :] = 0.05 * np.cos(2 * np.pi * high_freq * t) + np.random.normal(0, 0.01, points)
        imu[5, :] = np.random.normal(0, 0.01, points)

    elif platform_state == "high_vibration":
        # Rugged field ruts / drone wind turbulence: strong erratic shocks
        for ch in range(3):
            # Accel shock spikes
            shocks = np.random.choice([0, 1], size=points, p=[0.85, 0.15]) * np.random.uniform(-4.5, 4.5, points)
            noise = np.random.normal(0, 1.2, points)
            imu[ch, :] += (shocks + noise)
        imu[2, :] += 9.81
        for ch in range(3, 6):
            # Gyro erratic angular velocity
            imu[ch, :] = np.random.normal(0, 0.45, points)

    return imu


def generate_static_features(
    row: int,
    col: int,
    origin_lat: float = DEFAULT_FARM_ORIGIN[0],
    origin_lon: float = DEFAULT_FARM_ORIGIN[1],
) -> np.ndarray:
    """
    Generates static GPS and Soil properties for zone (row, col).
    Returns [5] vector: [lat, lon, elevation_m, soil_ph, organic_carbon_gkg]
    """
    # 25m grid spacing in degrees (~0.00022 deg lat)
    lat = origin_lat + row * 0.00025 + np.random.normal(0, 0.00002)
    lon = origin_lon + col * 0.00025 + np.random.normal(0, 0.00002)
    elevation = 210.0 + (row * 0.8) + (col * 0.5) + np.random.normal(0, 0.2)
    ph = 6.8 + np.random.normal(0, 0.3)
    organic_carbon = 18.5 + np.random.normal(0, 1.5)

    return np.array([lat, lon, elevation, ph, organic_carbon], dtype=np.float32)


def generate_forecasting_labels(
    stress_idx: int,
    severity: float,
) -> Tuple[int, int]:
    """
    Generates realistic forward stress labels at t+3 and t+7 days.
    Epidemiological transition:
    - Mild water stress can recover if irrigated (or progress to severe if drought continues).
    - Disease stress tends to escalate if untreated.
    - Healthy can stay healthy or rarely develop stress.
    """
    if stress_idx == 0:  # Healthy
        fc_3d = 0 if random.random() < 0.90 else random.choice([1, 4, 5])
        fc_7d = fc_3d if random.random() < 0.85 else random.choice([1, 2, 4])
    elif stress_idx == 1:  # Water stress
        # If mild (<40), high chance of rain/irrigation recovery; if severe (>70), persists
        if severity < 40 and random.random() < 0.35:
            fc_3d = 0  # recovered
        else:
            fc_3d = 1
        fc_7d = fc_3d if random.random() < 0.70 else random.choice([1, 2])
    elif stress_idx == 4:  # Disease stress
        # Highly persistent and progressive
        fc_3d = 4
        fc_7d = 4
    else:
        fc_3d = stress_idx if random.random() < 0.80 else random.choice([stress_idx, 0])
        fc_7d = fc_3d if random.random() < 0.75 else random.choice([stress_idx, 0])

    return fc_3d, fc_7d


def create_full_paired_dataset(
    output_dir: str,
    num_samples: int = 160,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Creates a paired multimodal dataset and saves it to output_dir.
    Creates sample images, env arrays, imu arrays, static features, and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []
    env_list = []
    imu_list = []
    static_list = []
    targets_cls = []
    targets_sev = []
    targets_fc3 = []
    targets_fc7 = []
    targets_qual = []
    targets_state = []
    zone_ids = []

    print(f"Generating {num_samples} multimodal crop stress samples in {output_dir}...")

    for i in range(num_samples):
        # Distribute stress classes evenly
        stress_idx = i % NUM_CLASSES
        # Realistic severity distribution (0 for healthy, 20-100 for stresses)
        if stress_idx == 0:
            severity = float(np.random.uniform(0.0, 10.0))
        else:
            severity = float(np.random.uniform(25.0, 95.0))

        # Platform motion state (cruising most common)
        platform_state = random.choices(
            PLATFORM_STATES,
            weights=[0.60, 0.15, 0.10, 0.15],
            k=1
        )[0]
        state_idx = PLATFORM_STATES.index(platform_state)

        # Field zone assignment (e.g. Zone A1 to D4)
        r = (i // FIELD_GRID_COLS) % FIELD_GRID_ROWS
        c = i % FIELD_GRID_COLS
        zone_name = f"Zone_{chr(65 + r)}{c + 1}"

        # 1. Image & Blur Quality Score
        img_np, quality_score = generate_synthetic_leaf_image(
            stress_idx=stress_idx,
            severity=severity,
            platform_state=platform_state,
            img_size=224,
        )
        img_filename = f"sample_{i:05d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # 2. Environmental 14-day sequence [14, 7]
        env_seq = generate_environmental_sequence(stress_idx, severity)

        # 3. IMU 6-channel waveform [6, 100]
        imu_wave = generate_imu_window(platform_state)

        # 4. Static features [5]
        static_feat = generate_static_features(r, c)

        # 5. Forecasting labels
        fc_3d, fc_7d = generate_forecasting_labels(stress_idx, severity)

        metadata.append({
            "sample_id": i,
            "image_file": img_filename,
            "zone_id": zone_name,
            "grid_row": r,
            "grid_col": c,
            "stress_class_idx": stress_idx,
            "stress_class_name": STRESS_CLASSES[stress_idx],
            "severity": round(severity, 2),
            "forecast_3d": fc_3d,
            "forecast_7d": fc_7d,
            "platform_state": platform_state,
            "platform_state_idx": state_idx,
            "blur_quality_score": round(quality_score, 4),
        })

        env_list.append(env_seq)
        imu_list.append(imu_wave)
        static_list.append(static_feat)
        targets_cls.append(stress_idx)
        targets_sev.append(severity)
        targets_fc3.append(fc_3d)
        targets_fc7.append(fc_7d)
        targets_qual.append(quality_score)
        targets_state.append(state_idx)
        zone_ids.append((r, c))

    # Save arrays to disk
    np.save(os.path.join(output_dir, "env_sequences.npy"), np.array(env_list, dtype=np.float32))
    np.save(os.path.join(output_dir, "imu_windows.npy"), np.array(imu_list, dtype=np.float32))
    np.save(os.path.join(output_dir, "static_features.npy"), np.array(static_list, dtype=np.float32))
    np.save(os.path.join(output_dir, "targets_cls.npy"), np.array(targets_cls, dtype=np.int64))
    np.save(os.path.join(output_dir, "targets_sev.npy"), np.array(targets_sev, dtype=np.float32))
    np.save(os.path.join(output_dir, "targets_fc3.npy"), np.array(targets_fc3, dtype=np.int64))
    np.save(os.path.join(output_dir, "targets_fc7.npy"), np.array(targets_fc7, dtype=np.int64))
    np.save(os.path.join(output_dir, "targets_qual.npy"), np.array(targets_qual, dtype=np.float32))
    np.save(os.path.join(output_dir, "targets_state.npy"), np.array(targets_state, dtype=np.int64))

    meta_file = os.path.join(output_dir, "metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "num_samples": num_samples,
        "classes": STRESS_CLASSES,
        "platform_states": PLATFORM_STATES,
        "output_dir": output_dir,
        "image_count": len(metadata),
    }

    print(f"Dataset successfully generated at {output_dir}. Total samples: {num_samples}")
    return summary


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
    create_full_paired_dataset(out_path, num_samples=120)

