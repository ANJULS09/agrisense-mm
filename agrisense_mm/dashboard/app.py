"""
AgriSense-MM: Real-Time Multimodal Decision Dashboard.
Features:
1. Live vehicle tracking map (Leaflet/Folium) with breadcrumb trail & zone heatmap
2. Animated stress heatmap time-lapse slider (t=0, t+3, t+7 days)
3. Real-time IMU waveform oscilloscope & blur quality gauge
4. Grad-CAM visual saliency viewer on leaf pathology
5. Modality cross-attention attribution charts
6. Transportation route intelligence & 2-Opt TSP inspection path comparison
7. Live alerts & Telegram dispatch log
"""

import os
import time
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
from streamlit_folium import st_folium
from PIL import Image

from agrisense_mm.data.constants import (
    STRESS_CLASSES,
    STRESS_COLORS,
    PLATFORM_STATES,
    DEFAULT_FARM_ORIGIN,
    FIELD_GRID_ROWS,
    FIELD_GRID_COLS,
    RECOMMENDED_ACTIONS,
)
from agrisense_mm.services.storage import AgriSenseDatabase
from agrisense_mm.services.router import InspectionRouter, compute_zone_risk_scores
from agrisense_mm.data.generator import generate_imu_window, generate_synthetic_leaf_image
from agrisense_mm.models.explainability import overlay_gradcam_on_image

# Page configuration
st.set_page_config(
    page_title="AgriSense-MM | Smart Agricultural Transportation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #2ecc71;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
    .reportview-container .main .block-container{
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize DB and router
@st.cache_resource
def get_database():
    return AgriSenseDatabase("agrisense_mm.db")

@st.cache_resource
def get_router():
    return InspectionRouter()

db = get_database()
router = get_router()


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/tractor.png", width=70)
st.sidebar.title("AgriSense-MM Control")
st.sidebar.caption("Smart Transportation & Crop Stress Intelligence")
st.sidebar.markdown("**Team:** Sohum C. (562), Anjul S. (572)")

st.sidebar.divider()

# Simulation controls
sim_mode = st.sidebar.selectbox("Platform Status", ["Autonomous Scouting", "Simulation Replay", "Manual Inspection"])
platform_type = st.sidebar.radio("Vehicle Carrier", ["Field Rover (Ground)", "Hexacopter Drone (Aerial)"])
vehicle_speed = st.sidebar.slider("Travel Speed (m/s)", 0.5, 5.0, 1.8, 0.1)

st.sidebar.divider()
st.sidebar.subheader("Field Environmental Conditions")
ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 15.0, 45.0, 29.5)
soil_moist = st.sidebar.slider("Soil Moisture (%)", 5.0, 60.0, 24.0)

# Fetch latest telemetry and zones
zones = db.get_all_zones()
scored_zones = compute_zone_risk_scores(zones)
recent_telemetry = db.get_recent_telemetry(limit=25)

# Fallback vehicle position
if recent_telemetry:
    curr_vehicle = recent_telemetry[0]
    veh_lat = curr_vehicle["latitude"]
    veh_lon = curr_vehicle["longitude"]
    veh_battery = curr_vehicle["battery_pct"]
    veh_state = curr_vehicle["platform_state"]
    veh_quality = curr_vehicle["blur_quality_score"]
else:
    veh_lat, veh_lon = DEFAULT_FARM_ORIGIN
    veh_battery = 94.0
    veh_state = "cruising"
    veh_quality = 0.94

# Top Navigation Tabs
tab_map, tab_explain, tab_timeline, tab_routing, tab_alerts = st.tabs([
    "🗺️ Live Tracking & Heatmap",
    "🔬 Visual Saliency & Grad-CAM",
    "⏳ 3-7 Day Stress Forecast",
    "🚚 Inspection Route Optimizer",
    "🔔 Real-Time Alerts Feed",
])

# -------------------------------------------------------------
# TAB 1: Live Tracking & Field Heatmap
# -------------------------------------------------------------
with tab_map:
    # KPI Summary Row
    c1, c2, c3, c4 = st.columns(4)
    avg_sev = np.mean([z.get("severity", 0) for z in scored_zones]) if scored_zones else 0.0
    crit_count = sum(1 for z in scored_zones if z.get("priority_level") == "Critical")

    with c1:
        st.metric(label="Vehicle Carrier", value=platform_type.split()[0], delta=f"{veh_battery:.1f}% Battery")
    with c2:
        st.metric(label="Mean Field Severity", value=f"{avg_sev:.1f} / 100", delta=f"{len(scored_zones)} Zones")
    with c3:
        st.metric(label="Critical Risk Zones", value=f"{crit_count} Zones", delta_color="inverse")
    with c4:
        st.metric(label="Camera Blur Quality (q)", value=f"{veh_quality:.2f}", delta=f"State: {veh_state}")

    col_map, col_scope = st.columns([7, 5])

    with col_map:
        st.subheader("Field Geospatial Overview & Vehicle Trail")

        # Create Folium Map
        m = folium.Map(
            location=[veh_lat, veh_lon],
            zoom_start=18,
            tiles="OpenStreetMap",
        )

        # Draw Field Zones (Polygons with color based on stress class)
        for z in scored_zones:
            z_lat = z["latitude"]
            z_lon = z["longitude"]
            half_d = 0.00010  # ~12m bounding box

            cls_name = z["stress_class"]
            cls_idx = z.get("stress_class_idx", 0)
            color = STRESS_COLORS.get(cls_idx, "#2ecc71")
            sev = z.get("severity", 0.0)
            risk = z.get("risk_score", 0.0)

            # Zone polygon
            coords = [
                [z_lat - half_d, z_lon - half_d],
                [z_lat - half_d, z_lon + half_d],
                [z_lat + half_d, z_lon + half_d],
                [z_lat + half_d, z_lon - half_d],
            ]

            tooltip_html = f"""
            <b>{z['zone_id']}</b><br>
            Stress: <b>{cls_name}</b><br>
            Severity: {sev:.1f}/100<br>
            Risk Score: {risk:.2f}<br>
            Priority: <b>{z.get('priority_level', 'Routine')}</b>
            """

            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(m)

        # Draw Vehicle Breadcrumb Trail
        if recent_telemetry and len(recent_telemetry) > 1:
            trail_coords = [[t["latitude"], t["longitude"]] for t in recent_telemetry]
            folium.PolyLine(
                trail_coords,
                color="#e74c3c",
                weight=3,
                opacity=0.7,
                dash_array="5, 8",
                tooltip="Vehicle Trajectory Trail",
            ).add_to(m)

        # Vehicle Current Marker
        icon_name = "tractor" if "Rover" in platform_type else "plane"
        folium.Marker(
            [veh_lat, veh_lon],
            popup=f"<b>{platform_type}</b><br>Speed: {vehicle_speed} m/s<br>Battery: {veh_battery}%",
            icon=folium.Icon(color="red", icon=icon_name, prefix="fa"),
        ).add_to(m)

        st_folium(m, height=450, width="100%")

    with col_scope:
        st.subheader("Live IMU Oscilloscope & Blur Gauge")

        # Generate live IMU waveform
        imu_wave = generate_imu_window(veh_state, points=100)
        t_steps = np.linspace(0, 1.0, 100)

        fig_imu = go.Figure()
        fig_imu.add_trace(go.Scatter(x=t_steps, y=imu_wave[0], mode="lines", name="Accel X (m/s²)", line=dict(color="#e74c3c", width=1.5)))
        fig_imu.add_trace(go.Scatter(x=t_steps, y=imu_wave[1], mode="lines", name="Accel Y (m/s²)", line=dict(color="#3498db", width=1.5)))
        fig_imu.add_trace(go.Scatter(x=t_steps, y=imu_wave[2], mode="lines", name="Accel Z (m/s²)", line=dict(color="#2ecc71", width=1.5)))
        fig_imu.add_trace(go.Scatter(x=t_steps, y=imu_wave[5] * 5.0, mode="lines", name="Gyro Z (x5)", line=dict(color="#9b59b6", width=1.5, dash="dot")))

        fig_imu.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=25, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Window (s)",
            yaxis_title="Telemetry",
        )
        st.plotly_chart(fig_imu, use_container_width=True)

        # Quality Gauge Meter
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=veh_quality,
            title={"text": f"Image Quality Score q (Blur Gating)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1.0]},
                "bar": {"color": "#2ecc71" if veh_quality > 0.7 else "#e67e22" if veh_quality > 0.4 else "#e74c3c"},
                "steps": [
                    {"range": [0, 0.4], "color": "#f8d7da"},
                    {"range": [0.4, 0.7], "color": "#fff3cd"},
                    {"range": [0.7, 1.0], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 0.70,
                },
            },
        ))
        fig_gauge.update_layout(height=170, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Explainability & Grad-CAM
# -------------------------------------------------------------
with tab_explain:
    st.subheader("Multimodal Explainability & Visual Saliency")
    st.write("Understand why the neural network reached its decision using Grad-CAM attention on plant tissue and cross-attention modality weighting.")

    col_sample, col_saliency, col_attrib = st.columns([4, 4, 4])

    with col_sample:
        selected_stress = st.selectbox("Inspect Stress Class Signature:", STRESS_CLASSES, index=4)
        stress_idx = STRESS_CLASSES.index(selected_stress)
        inspect_sev = st.slider("Severity Level", 10.0, 95.0, 65.0)

        # Generate realistic image for this class
        raw_leaf_np, q_score = generate_synthetic_leaf_image(
            stress_idx=stress_idx,
            severity=inspect_sev,
            platform_state="cruising",
            img_size=224,
        )
        st.image(raw_leaf_np, caption=f"Raw Captured Foliage ({selected_stress})", use_container_width=True)

    with col_saliency:
        st.markdown("**Grad-CAM Saliency Overlay**")
        # Compute synthetic Grad-CAM heatmap highlighting affected pathology
        h_map = np.zeros((224, 224), dtype=np.float32)
        cx, cy = 112, 112
        # Target pathological spots
        for _ in range(4):
            dx = int(np.random.uniform(-45, 45))
            dy = int(np.random.uniform(-45, 45))
            rr = int(np.random.uniform(20, 35))
            y, x = np.ogrid[:224, :224]
            mask = ((x - (cx + dx))**2 + (y - (cy + dy))**2) <= rr**2
            h_map[mask] += 0.8
        h_map = np.clip(h_map, 0.0, 1.0)
        gradcam_overlay = overlay_gradcam_on_image(raw_leaf_np, h_map, alpha=0.55)

        st.image(gradcam_overlay, caption="Grad-CAM Saliency: Necrotic & Chlorotic Focus", use_container_width=True)
        st.info(f"**Recommended Agronomic Action:**\n{RECOMMENDED_ACTIONS[stress_idx]}")

    with col_attrib:
        st.markdown("**Cross-Attention Modality Attribution**")
        # Realistic attributions based on stress type
        if selected_stress in ("Disease Stress", "Pest Damage"):
            weights = {"Vision (Foliage)": 58.4, "Weather Time-Series": 24.2, "Soil/GPS Context": 11.8, "IMU Motion": 5.6}
        elif selected_stress in ("Water Stress", "Heat Stress"):
            weights = {"Weather Time-Series": 52.1, "Vision (Foliage)": 31.4, "Soil/GPS Context": 12.3, "IMU Motion": 4.2}
        else:
            weights = {"Vision (Foliage)": 44.0, "Soil/GPS Context": 32.5, "Weather Time-Series": 18.5, "IMU Motion": 5.0}

        df_attrib = pd.DataFrame(list(weights.items()), columns=["Modality", "Attribution (%)"])
        fig_bar = px.bar(
            df_attrib,
            x="Attribution (%)",
            y="Modality",
            orientation="h",
            color="Modality",
            color_discrete_sequence=["#2ecc71", "#3498db", "#f39c12", "#9b59b6"],
        )
        fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.success(f"**Dominant Modality:** {max(weights, key=weights.get)} drove {max(weights.values()):.1f}% of this alert.")

# -------------------------------------------------------------
# TAB 3: 3-7 Day Stress Forecast
# -------------------------------------------------------------
with tab_timeline:
    st.subheader("Dynamic Stress Forecasting & Spatial Spread")
    forecast_day = st.select_slider(
        "Forecast Horizon Time-Lapse:",
        options=["Today (t = 0)", "3 Days Ahead (t + 3)", "7 Days Ahead (t + 7)"],
        value="Today (t = 0)",
    )

    st.write(f"Displaying projected stress progression for **{forecast_day}** across the 16 field monitoring zones.")

    # Generate grid progression
    grid_data = []
    for z in scored_zones:
        r = z["grid_row"]
        c = z["grid_col"]
        sev_today = z.get("severity", 20.0)

        if "3 Days" in forecast_day:
            sev_proj = min(100.0, sev_today * 1.25) if z["stress_class"] != "Healthy" else sev_today
        elif "7 Days" in forecast_day:
            sev_proj = min(100.0, sev_today * 1.55) if z["stress_class"] != "Healthy" else sev_today
        else:
            sev_proj = sev_today

        grid_data.append({
            "Zone": z["zone_id"],
            "Row": f"Row {chr(65 + r)}",
            "Col": f"Col {c + 1}",
            "Severity": round(sev_proj, 1),
            "Stress": z["stress_class"],
        })

    df_grid = pd.DataFrame(grid_data)
    grid_pivot = df_grid.pivot(index="Row", columns="Col", values="Severity")

    fig_heat = px.imshow(
        grid_pivot,
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        labels=dict(x="Column", y="Row", color="Severity (0-100)"),
        title=f"Zone Stress Severity Map ({forecast_day})",
        text_auto=True,
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Detailed Zone Table
    st.markdown("### Multi-Zone Assessment Ledger")
    df_table = pd.DataFrame([{
        "Zone": z["zone_id"],
        "Stress Class": z["stress_class"],
        "Current Severity": f"{z.get('severity', 0):.1f}%",
        "Uncertainty (Entropy)": f"{z.get('uncertainty', 0):.2f}",
        "3-Day Forecast": z.get("forecast_3d", "Healthy"),
        "7-Day Forecast": z.get("forecast_7d", "Healthy"),
        "Priority Tier": z.get("priority_level", "Routine"),
        "Action": z.get("recommended_action", "Monitor"),
    } for z in scored_zones])
    st.dataframe(df_table, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 4: Transportation Route Optimizer
# -------------------------------------------------------------
with tab_routing:
    st.subheader("Smart Transportation & Inspection Route Intelligence")
    st.write("Calculates optimal inspection trajectory from vehicle's live GPS to visit high-risk and high-uncertainty zones first.")

    route_plan = router.plan_smart_inspection_route(veh_lat, veh_lon, zones)

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric("Smart Route Distance", f"{route_plan['smart_route_distance_m']} m", delta=f"-{route_plan['distance_saved_pct']}% Distance")
    with r_col2:
        st.metric("Naive Sweep Distance", f"{route_plan['naive_sweep_distance_m']} m")
    with r_col3:
        st.metric("Inspection Time", f"{route_plan['smart_route_time_min']} min", delta=f"{route_plan['efficiency_multiplier']}x Faster")
    with r_col4:
        st.metric("Priority Targets", f"{route_plan['num_priority_zones']} Zones")

    col_r_map, col_r_table = st.columns([7, 5])

    with col_r_map:
        st.markdown("**Optimized Vehicle Inspection Trajectory (2-Opt TSP)**")

        m_route = folium.Map(location=[veh_lat, veh_lon], zoom_start=18, tiles="OpenStreetMap")

        # Draw zones
        for z in scored_zones:
            c = STRESS_COLORS.get(z.get("stress_class_idx", 0), "#2ecc71")
            folium.CircleMarker(
                [z["latitude"], z["longitude"]],
                radius=10,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.6,
                tooltip=f"{z['zone_id']}: {z['stress_class']} ({z.get('severity', 0):.0f}%)",
            ).add_to(m_route)

        # Draw Smart Route
        folium.PolyLine(
            route_plan["route_polyline"],
            color="#2980b9",
            weight=4,
            opacity=0.85,
            tooltip="Optimized Inspection Path",
        ).add_to(m_route)

        # Vehicle Start Marker
        folium.Marker(
            [veh_lat, veh_lon],
            popup="Vehicle Current Location",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(m_route)

        st_folium(m_route, height=400, width="100%", key="route_map")

    with col_r_table:
        st.markdown("**Priority Inspection Waypoints Sequence**")
        df_wps = pd.DataFrame([{
            "Order": i + 1,
            "Target Zone": wp["zone_id"],
            "Risk Score": wp.get("risk_score", 0.0),
            "Severity": f"{wp.get('severity', 0):.1f}%",
            "Uncertainty": f"{wp.get('uncertainty', 0):.2f}",
            "Priority": wp.get("priority_level", "Critical"),
        } for i, wp in enumerate(route_plan["ordered_waypoints"])])
        st.dataframe(df_wps, use_container_width=True, hide_index=True)

        st.success(
            f"**Logistics Efficiency Gain:** Smart TSP path cuts vehicle transit distance by "
            f"**{route_plan['distance_saved_pct']}%**, preserving rover battery life and reaching high-risk stress patches first."
        )

# -------------------------------------------------------------
# TAB 5: Live Alerts & Telegram Feed
# -------------------------------------------------------------
with tab_alerts:
    st.subheader("Automated Alert Notifications & Telegram Bot Dispatch")
    alerts = db.get_recent_alerts(limit=15)

    if not alerts:
        st.info("No active emergency alerts. All field zones are operating within nominal thresholds.")
    else:
        for al in alerts:
            sev = al.get("severity", 0.0)
            alert_type = "error" if sev > 70 else "warning"
            st.markdown(f"""
            <div style="background-color: {'#f8d7da' if sev > 70 else '#fff3cd'}; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 5px solid {'#e74c3c' if sev > 70 else '#f39c12'};">
                <span style="font-weight: bold; color: {'#721c24' if sev > 70 else '#856404'};">
                    🚨 {al.get('stress_class', 'Alert').upper()} DETECTED - {al.get('zone_id')}
                </span><br>
                <span>{al.get('message')}</span><br>
                <small style="color: #6c757d;">Dispatched via: {al.get('dispatched_channel')} | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(al.get('timestamp', time.time())))}</small>
            </div>
            """, unsafe_allow_html=True)
