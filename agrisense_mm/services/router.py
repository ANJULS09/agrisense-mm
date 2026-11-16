"""
AgriSense-MM: Transportation & Inspection Route Intelligence Engine.
Focuses on mobile sensing logistics:
1. Zone Risk Index = alpha * (Severity/100) + beta * Uncertainty + gamma * Contagion
2. High-priority inspection target selection
3. 2-Opt Traveling Salesperson (TSP) / A* path planning from vehicle's live GPS
4. Quantitative route benchmark vs naive boustrophedon (sweep) path
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from agrisense_mm.data.constants import DEFAULT_FARM_ORIGIN, FIELD_GRID_ROWS, FIELD_GRID_COLS


def haversine_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Computes great-circle ground distance in meters between two GPS coordinates."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_zone_risk_scores(
    zones: List[Dict[str, Any]],
    alpha_sev: float = 0.50,
    beta_unc: float = 0.35,
    gamma_contagion: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Computes unified composite risk score for each field zone:
    Risk(z) = alpha * (Severity/100) + beta * Uncertainty + gamma * ContagionRisk
    """
    # Create lookup map
    zone_map = {z["zone_id"]: z for z in zones}

    # Helper to calculate contagion from adjacent neighbors
    for z in zones:
        r = z.get("grid_row", 0)
        c = z.get("grid_col", 0)
        neighbor_sevs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor_id = f"Zone_{chr(65 + nr)}{nc + 1}"
            if neighbor_id in zone_map:
                neighbor_sevs.append(zone_map[neighbor_id].get("severity", 0.0))

        contagion = float(np.mean(neighbor_sevs) / 100.0) if neighbor_sevs else 0.0

        sev_norm = float(z.get("severity", 0.0)) / 100.0
        unc = float(z.get("uncertainty", 0.0))

        risk = (alpha_sev * sev_norm) + (beta_unc * unc) + (gamma_contagion * contagion)
        z["risk_score"] = round(float(np.clip(risk, 0.0, 1.0)), 4)
        z["contagion_risk"] = round(contagion, 3)

        # Priority category
        if z["risk_score"] >= 0.55 or z.get("severity", 0) >= 65:
            z["priority_level"] = "Critical"
        elif z["risk_score"] >= 0.35 or z.get("severity", 0) >= 35:
            z["priority_level"] = "High"
        else:
            z["priority_level"] = "Routine"

    return zones


def solve_2opt_tsp(
    start_pos: Tuple[float, float],
    waypoints: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Solves Traveling Salesperson Problem (TSP) using Nearest Neighbor + 2-Opt local refinement.
    Minimizes total vehicle battery / travel distance starting from current GPS.
    """
    if len(waypoints) <= 1:
        total_dist = 0.0
        if waypoints:
            total_dist = haversine_distance_meters(start_pos[0], start_pos[1], waypoints[0]["latitude"], waypoints[0]["longitude"])
        return waypoints, total_dist

    # 1. Greedy Nearest Neighbor initial tour
    unvisited = list(waypoints)
    current_lat, current_lon = start_pos
    tour = []

    while unvisited:
        nearest_idx = min(
            range(len(unvisited)),
            key=lambda i: haversine_distance_meters(
                current_lat, current_lon, unvisited[i]["latitude"], unvisited[i]["longitude"]
            ),
        )
        nxt = unvisited.pop(nearest_idx)
        tour.append(nxt)
        current_lat, current_lon = nxt["latitude"], nxt["longitude"]

    # 2. 2-Opt heuristic optimization
    def calculate_tour_distance(t: List[Dict[str, Any]]) -> float:
        d = haversine_distance_meters(start_pos[0], start_pos[1], t[0]["latitude"], t[0]["longitude"])
        for i in range(len(t) - 1):
            d += haversine_distance_meters(t[i]["latitude"], t[i]["longitude"], t[i+1]["latitude"], t[i+1]["longitude"])
        return d

    improved = True
    best_dist = calculate_tour_distance(tour)
    iterations = 0

    while improved and iterations < 50:
        improved = False
        iterations += 1
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                # Reverse segment between i and j
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_dist = calculate_tour_distance(new_tour)
                if new_dist < best_dist - 0.5:
                    tour = new_tour
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break

    return tour, best_dist


class InspectionRouter:
    def __init__(
        self,
        vehicle_speed_mps: float = 2.0,  # 2 m/s rover/drone scouting speed
    ):
        self.vehicle_speed_mps = vehicle_speed_mps

    def plan_smart_inspection_route(
        self,
        current_lat: float,
        current_lon: float,
        zones: List[Dict[str, Any]],
        max_waypoints: int = 8,
    ) -> Dict[str, Any]:
        """
        Generates risk-optimized inspection route from vehicle's live GPS.
        Compares efficiency against naive boustrophedon sweep.
        """
        scored_zones = compute_zone_risk_scores(zones)

        # Filter high-priority inspection targets (Critical & High, or top-k by risk)
        priority_zones = [z for z in scored_zones if z["priority_level"] in ("Critical", "High")]
        if len(priority_zones) < 3:
            # Fallback to top-k risk
            sorted_all = sorted(scored_zones, key=lambda z: z["risk_score"], reverse=True)
            priority_zones = sorted_all[:max_waypoints]
        else:
            priority_zones = sorted(priority_zones, key=lambda z: z["risk_score"], reverse=True)[:max_waypoints]

        # Solve TSP on priority targets
        start_coord = (current_lat, current_lon)
        optimized_tour, smart_distance = solve_2opt_tsp(start_coord, priority_zones)

        # Benchmark: Naive boustrophedon (row-by-row zigzag sweep across all zones)
        sweep_zones = sorted(zones, key=lambda z: (z.get("grid_row", 0), z.get("grid_col", 0) if z.get("grid_row", 0) % 2 == 0 else -z.get("grid_col", 0)))
        sweep_distance = 0.0
        cur_lat, cur_lon = current_lat, current_lon
        for sz in sweep_zones:
            sweep_distance += haversine_distance_meters(cur_lat, cur_lon, sz["latitude"], sz["longitude"])
            cur_lat, cur_lon = sz["latitude"], sz["longitude"]

        # Logistics Metrics
        smart_time_sec = smart_distance / self.vehicle_speed_mps
        sweep_time_sec = sweep_distance / self.vehicle_speed_mps
        dist_saved_pct = max(0.0, ((sweep_distance - smart_distance) / max(1.0, sweep_distance)) * 100.0)

        # Build polyline coordinate track for map visualization
        route_polyline = [[current_lat, current_lon]]
        for wp in optimized_tour:
            route_polyline.append([wp["latitude"], wp["longitude"]])

        return {
            "current_vehicle_pos": [current_lat, current_lon],
            "num_priority_zones": len(optimized_tour),
            "smart_route_distance_m": round(smart_distance, 1),
            "smart_route_time_min": round(smart_time_sec / 60.0, 1),
            "naive_sweep_distance_m": round(sweep_distance, 1),
            "naive_sweep_time_min": round(sweep_time_sec / 60.0, 1),
            "distance_saved_pct": round(dist_saved_pct, 1),
            "efficiency_multiplier": round(sweep_distance / max(1.0, smart_distance), 2),
            "ordered_waypoints": optimized_tour,
            "route_polyline": route_polyline,
        }

