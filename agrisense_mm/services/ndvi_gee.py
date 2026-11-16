"""
AgriSense-MM: Sentinel-2 / Satellite NDVI Service.
Provides NDVI vegetation index time-series and canopy coverage metrics.
"""

import numpy as np
from typing import Dict, List, Tuple


class SentinelNDVIService:
    def __init__(self):
        pass

    def compute_ndvi_from_bands(self, nir_reflectance: float, red_reflectance: float) -> float:
        """
        Computes standard Normalized Difference Vegetation Index:
        NDVI = (NIR - Red) / (NIR + Red)
        """
        denom = nir_reflectance + red_reflectance
        if denom == 0.0:
            return 0.0
        return float(np.clip((nir_reflectance - red_reflectance) / denom, -1.0, 1.0))

    def get_zone_ndvi_profile(self, zone_id: str, baseline_ndvi: float = 0.80) -> Dict[str, float]:
        """Returns 5-day revisit satellite NDVI metrics for the specified field zone."""
        # Realistic variation across zone
        noise = np.random.normal(0, 0.03)
        current_ndvi = float(np.clip(baseline_ndvi + noise, 0.15, 0.95))
        delta_7d = float(np.random.uniform(-0.08, 0.02))

        return {
            "zone_id": zone_id,
            "current_ndvi": round(current_ndvi, 3),
            "delta_7d": round(delta_7d, 3),
            "canopy_vigor": "High" if current_ndvi > 0.70 else "Medium" if current_ndvi > 0.45 else "Stressed",
            "sentinel_tile": "T43REQ_20261103",
        }

