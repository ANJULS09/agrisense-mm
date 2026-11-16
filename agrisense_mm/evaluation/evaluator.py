"""
AgriSense-MM: Comprehensive Evaluation and Ablation Runner.
Orchestrates:
1. Ablation Study:
   - Config A: Vision Only
   - Config B: Vision + Environmental
   - Config C: Vision + Environmental + Motion
   - Config D: Full AgriSense-MM (with Cross-Attention + Spatial GNN)
2. Missing Modality Robustness Stress Tests
3. Motion Blur Degradation Analysis
4. End-to-end Inference Latency Benchmarks
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional

from agrisense_mm.models.agrisense_net import AgriSenseMMNet
from agrisense_mm.training.dataset import create_dataloaders
from agrisense_mm.evaluation.metrics import evaluate_multitask_predictions


def run_model_inference_on_loader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_modality: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Runs forward inference over dataloader and aggregates all outputs."""
    model.eval()

    all_preds_cls = []
    all_probs_cls = []
    all_targets_cls = []
    all_preds_sev = []
    all_targets_sev = []
    all_preds_fc3 = []
    all_targets_fc3 = []
    all_preds_fc7 = []
    all_targets_fc7 = []
    all_latencies = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            env_seqs = batch["env_seqs"].to(device)
            imu_windows = batch["imu_windows"].to(device)
            static_feats = batch["static_feats"].to(device)
            edge_index = batch["edge_index"].to(device)

            # Apply simulated missing modality if requested
            if mask_modality == "drop_vision":
                images = torch.zeros_like(images)
            elif mask_modality == "drop_env":
                env_seqs = torch.zeros_like(env_seqs)
            elif mask_modality == "drop_motion":
                imu_windows = torch.zeros_like(imu_windows)

            t0 = time.perf_counter()
            outputs = model(images, env_seqs, imu_windows, static_feats, edge_index)
            lat_ms = (time.perf_counter() - t0) * 1000.0 / images.size(0)
            all_latencies.append(lat_ms)

            probs_cls = torch.softmax(outputs["cls_logits"], dim=-1).cpu().numpy()
            preds_cls = np.argmax(probs_cls, axis=-1)

            all_preds_cls.extend(preds_cls)
            all_probs_cls.extend(probs_cls)
            all_targets_cls.extend(batch["targets_cls"].numpy())

            all_preds_sev.extend(outputs["severity"].cpu().numpy().flatten())
            all_targets_sev.extend(batch["targets_sev"].numpy().flatten())

            all_preds_fc3.extend(torch.argmax(outputs["forecast_3d_logits"], dim=-1).cpu().numpy())
            all_targets_fc3.extend(batch["targets_fc3"].numpy())

            all_preds_fc7.extend(torch.argmax(outputs["forecast_7d_logits"], dim=-1).cpu().numpy())
            all_targets_fc7.extend(batch["targets_fc7"].numpy())

    return {
        "y_true_cls": np.array(all_targets_cls),
        "y_pred_cls": np.array(all_preds_cls),
        "y_probs_cls": np.array(all_probs_cls),
        "y_true_sev": np.array(all_targets_sev),
        "y_pred_sev": np.array(all_preds_sev),
        "y_true_fc3": np.array(all_targets_fc3),
        "y_pred_fc3": np.array(all_preds_fc3),
        "y_true_fc7": np.array(all_targets_fc7),
        "y_pred_fc7": np.array(all_preds_fc7),
        "avg_latency_ms": float(np.mean(all_latencies)),
    }


def run_ablation_study(
    data_dir: str,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "evaluation/results",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates 4 architectural ablations on the exact same validation partition:
    1. Vision Only
    2. Vision + Environmental
    3. Vision + Environmental + Motion
    4. Full AgriSense-MM (Cross-Attention + Spatial GNN)
    """
    os.makedirs(output_dir, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _, val_loader = create_dataloaders(data_dir=data_dir, batch_size=8)

    ablation_configs = {
        "1_Vision_Only": {
            "enable_env": False,
            "enable_motion": False,
            "enable_gnn": False,
            "description": "Baseline: Leaf images only (EfficientNet-B0)",
        },
        "2_Vision_Plus_Env": {
            "enable_env": True,
            "enable_motion": False,
            "enable_gnn": False,
            "description": "Multimodal: Leaf images + 14-day Weather/Soil Bi-LSTM",
        },
        "3_Vision_Env_Motion": {
            "enable_env": True,
            "enable_motion": True,
            "enable_gnn": False,
            "description": "Vision + Env + 1D-CNN IMU Quality Blur Gating",
        },
        "4_Full_AgriSense_MM": {
            "enable_env": True,
            "enable_motion": True,
            "enable_gnn": True,
            "description": "Full System: Cross-Attention Fusion + Spatial Zone GNN",
        },
    }

    results = {}

    print("\n" + "=" * 70)
    print("RUNNING AGRISENSE-MM ABLATION STUDY")
    print("=" * 70)

    for config_name, cfg in ablation_configs.items():
        print(f"\n--> Evaluating Configuration: {config_name}")
        print(f"    Details: {cfg['description']}")

        model = AgriSenseMMNet(
            enable_env=cfg["enable_env"],
            enable_motion=cfg["enable_motion"],
            enable_gnn=cfg["enable_gnn"],
        ).to(device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=device)
            # Load matching weights
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state["model_state_dict"].items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        preds = run_model_inference_on_loader(model, val_loader, device)
        metrics = evaluate_multitask_predictions(
            y_true_cls=preds["y_true_cls"],
            y_pred_cls=preds["y_pred_cls"],
            y_probs_cls=preds["y_probs_cls"],
            y_true_sev=preds["y_true_sev"],
            y_pred_sev=preds["y_pred_sev"],
            y_true_fc3=preds["y_true_fc3"],
            y_pred_fc3=preds["y_pred_fc3"],
            y_true_fc7=preds["y_true_fc7"],
            y_pred_fc7=preds["y_pred_fc7"],
        )
        metrics["latency_ms_per_sample"] = round(preds["avg_latency_ms"], 2)
        metrics["description"] = cfg["description"]
        results[config_name] = metrics

        cls_m = metrics["classification"]
        sev_m = metrics["severity"]
        fc_m = metrics["forecasting"]
        print(
            f"    Result: Accuracy={cls_m['accuracy']:.4f} | Macro-F1={cls_m['f1_macro']:.4f} | "
            f"ECE={cls_m['ece']:.4f} | Sev-MAE={sev_m['mae']:.2f} | Latency={metrics['latency_ms_per_sample']}ms"
        )

    # Missing Modality Robustness Test on Full Model
    print("\n" + "=" * 70)
    print("RUNNING MISSING MODALITY ROBUSTNESS TEST")
    print("=" * 70)

    full_model = AgriSenseMMNet(enable_env=True, enable_motion=True, enable_gnn=True).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        full_model.load_state_dict(state["model_state_dict"])

    modality_scenarios = {
        "all_modalities_present": None,
        "camera_failure_drop_vision": "drop_vision",
        "weather_sensor_drop_env": "drop_env",
        "imu_failure_drop_motion": "drop_motion",
    }

    modality_results = {}
    for scenario_name, mask_flag in modality_scenarios.items():
        preds = run_model_inference_on_loader(full_model, val_loader, device, mask_modality=mask_flag)
        m = evaluate_multitask_predictions(
            y_true_cls=preds["y_true_cls"],
            y_pred_cls=preds["y_pred_cls"],
            y_probs_cls=preds["y_probs_cls"],
            y_true_sev=preds["y_true_sev"],
            y_pred_sev=preds["y_pred_sev"],
            y_true_fc3=preds["y_true_fc3"],
            y_pred_fc3=preds["y_pred_fc3"],
            y_true_fc7=preds["y_true_fc7"],
            y_pred_fc7=preds["y_pred_fc7"],
        )
        modality_results[scenario_name] = {
            "accuracy": m["classification"]["accuracy"],
            "f1_macro": m["classification"]["f1_macro"],
            "sev_mae": m["severity"]["mae"],
        }
        print(f"    Scenario [{scenario_name}]: Acc={m['classification']['accuracy']:.4f}, F1={m['classification']['f1_macro']:.4f}")

    results["missing_modality_robustness"] = modality_results

    # Save to disk
    result_path = os.path.join(output_dir, "ablation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll ablation results successfully saved to {result_path}")
    return results


if __name__ == "__main__":
    run_ablation_study(
        data_dir="agrisense_mm/data/synthetic",
        checkpoint_path="checkpoints/best_agrisense_model.pt",
    )

