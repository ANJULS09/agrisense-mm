"""
AgriSense-MM: Master Orchestrator Script.
Runs the complete workflow:
1. Dataset generation / verification
2. Full test suite execution
3. Model training and checkpoint saving
4. Ablation study & benchmark evaluation
5. Plot generation
6. Database seeding with realistic multi-zone field state
"""

import os
import sys
import subprocess
import time

def run_step(step_name: str, cmd: list):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[!] Step '{step_name}' failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✓] {step_name} completed successfully in {elapsed:.2f}s")


def main():
    py_exec = sys.executable

    print("=" * 70)
    print("AGRISENSE-MM: MULTIMODAL CROP STRESS & SMART TRANSPORTATION")
    print("Authors: Sohum Chandra Srivastava (562), Anjul Shukla (572)")
    print("=" * 70)

    # Step 1: Synthesize Dataset
    data_dir = "agrisense_mm/data/synthetic"
    if not os.path.exists(os.path.join(data_dir, "metadata.json")):
        run_step("1. Multimodal Dataset Generation", [py_exec, "agrisense_mm/data/generator.py"])
    else:
        print("\n[✓] Synthetic paired dataset already exists.")

    # Step 2: Run Unit Tests
    run_step("2. Unit Test Suite (Models, Gating, Routing, Pipeline)", [py_exec, "-m", "pytest", "agrisense_mm/tests", "-v"])

    # Step 3: Train Model Checkpoint
    checkpoint_file = "checkpoints/best_agrisense_model.pt"
    if not os.path.exists(checkpoint_file):
        run_step("3. Model Training (5 Epochs with Cosine Annealing)", [py_exec, "agrisense_mm/training/train.py", "--epochs", "5"])
    else:
        print(f"\n[✓] Trained model checkpoint found at {checkpoint_file}.")

    # Step 4: Run Ablation Study & Robustness Benchmarks
    run_step("4. Ablation Study & Missing Modality Benchmarks", [py_exec, "agrisense_mm/evaluation/evaluator.py"])

    # Step 5: Generate Publication Figures
    run_step("5. Automated Figure Generation", [py_exec, "agrisense_mm/evaluation/plot_results.py"])

    # Step 6: Seed Database with Multi-Zone Farm Field
    print("\n" + "=" * 70)
    print("STEP: 6. Database Initialization & Seeding")
    print("=" * 70)
    from agrisense_mm.services.storage import AgriSenseDatabase
    from agrisense_mm.data.constants import DEFAULT_FARM_ORIGIN, FIELD_GRID_ROWS, FIELD_GRID_COLS, RECOMMENDED_ACTIONS, STRESS_CLASSES
    import random

    db = AgriSenseDatabase("agrisense_mm.db")
    # Seed 16 zones with realistic varied stress conditions
    sample_stresses = [0, 1, 4, 0, 2, 3, 5, 0, 1, 0, 4, 2, 0, 0, 3, 0]
    for r in range(FIELD_GRID_ROWS):
        for c in range(FIELD_GRID_COLS):
            idx = r * FIELD_GRID_COLS + c
            st_idx = sample_stresses[idx % len(sample_stresses)]
            st_name = STRESS_CLASSES[st_idx]
            sev = float(random.uniform(5.0, 15.0) if st_idx == 0 else random.uniform(35.0, 88.0))
            unc = float(random.uniform(0.05, 0.20) if st_idx == 0 else random.uniform(0.25, 0.65))
            zone_id = f"Zone_{chr(65 + r)}{c + 1}"
            lat = DEFAULT_FARM_ORIGIN[0] + r * 0.00025
            lon = DEFAULT_FARM_ORIGIN[1] + c * 0.00025

            db.upsert_zone_assessment({
                "zone_id": zone_id,
                "grid_row": r,
                "grid_col": c,
                "latitude": lat,
                "longitude": lon,
                "stress_class": st_name,
                "stress_class_idx": st_idx,
                "severity": round(sev, 1),
                "uncertainty": round(unc, 2),
                "forecast_3d": st_name if sev > 50 else "Healthy",
                "forecast_7d": st_name if sev > 60 else "Healthy",
                "risk_score": round((sev/100.0) * 0.6 + unc * 0.4, 2),
                "last_assessed_timestamp": time.time(),
                "recommended_action": RECOMMENDED_ACTIONS[st_idx],
            })

            if sev > 65.0:
                db.log_alert({
                    "zone_id": zone_id,
                    "severity": sev,
                    "stress_class": st_name,
                    "message": f"Critical {st_name} outbreak in {zone_id}! Immediate scouting required.",
                    "dispatched_channel": "Telegram Bot & Dashboard",
                })

    # Log initial telemetry
    db.log_telemetry({
        "vehicle_id": "rover-01",
        "latitude": DEFAULT_FARM_ORIGIN[0],
        "longitude": DEFAULT_FARM_ORIGIN[1],
        "speed_mps": 1.8,
        "battery_pct": 95.0,
        "platform_state": "cruising",
        "blur_quality_score": 0.94,
        "imu_variance": 0.11,
    })
    print("[✓] Farm database seeded with 16 monitoring zones and initial rover telemetry.")

    print("\n" + "=" * 70)
    print("ALL WORKFLOW STEPS COMPLETE!")
    print("=" * 70)
    print("To start the services:")
    print("  1. Backend API:      python serve.py")
    print("  2. Streamlit UI:     streamlit run dashboard/app.py")
    print("  3. Sensor Simulator: python simulate.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

