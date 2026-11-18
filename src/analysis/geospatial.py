"""Geospatial grid transformation and temporal disease spread modeling for TerraSpectra.

Provides georeferenced PyDeck 3D grid point generation from probability and spectral
rasters, as well as spatial diffusion modeling for multi-day disease progression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np


def raster_to_geogrid(
    prob_map: np.ndarray,
    ndvi_map: np.ndarray,
    stress_map: Optional[np.ndarray] = None,
    origin_lat: float = 18.5204,
    origin_lon: float = 73.8567,
    pixel_spacing_deg: float = 0.00015,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Convert 2D raster probability and index maps into geo-referenced 3D column records.

    Generates records formatted for PyDeck ColumnLayer and GeoJson visualization,
    mapping disease probabilities into spatial elevations and RGB risk palettes.

    Args:
        prob_map: 2D array of pixel-level fungal blight probability in [0.0, 1.0].
        ndvi_map: 2D array of NDVI vegetation index in [-1.0, 1.0].
        stress_map: Optional 2D array of Red-Edge Stress Index in [-1.0, 1.0].
        origin_lat: Reference origin latitude in decimal degrees (default: 18.5204).
        origin_lon: Reference origin longitude in decimal degrees (default: 73.8567).
        pixel_spacing_deg: Ground sampling distance in decimal degrees (default: 0.00015 deg ~ 15m).
        threshold: Decision threshold for classifying a pixel as infected (default: 0.5).

    Returns:
        List of dictionary records, each containing:
            - 'row': Grid row index
            - 'col': Grid column index
            - 'lat': Georeferenced latitude
            - 'lon': Georeferenced longitude
            - 'probability': Disease probability in [0.0, 1.0]
            - 'ndvi': NDVI value
            - 'stress': Red-Edge stress index (0.0 if stress_map is None)
            - 'elevation': Scaled column height in meters for 3D visualization
            - 'risk_label': 'Healthy', 'Watch', or 'Infected'
            - 'color_rgba': [r, g, b, a] color list

    Raises:
        ValueError: If input arrays are not 2D or spatial dimensions mismatch.
    """
    # Accept torch tensors if provided
    if hasattr(prob_map, "detach"):
        prob_map = prob_map.detach().cpu().numpy()
    if hasattr(ndvi_map, "detach"):
        ndvi_map = ndvi_map.detach().cpu().numpy()
    if stress_map is not None and hasattr(stress_map, "detach"):
        stress_map = stress_map.detach().cpu().numpy()

    if not isinstance(prob_map, np.ndarray) or not isinstance(ndvi_map, np.ndarray):
        raise TypeError("prob_map and ndvi_map must be numpy ndarrays.")

    if prob_map.ndim != 2:
        raise ValueError(f"prob_map must be a 2D array, got {prob_map.ndim}D shape {prob_map.shape}")
    if ndvi_map.ndim != 2:
        raise ValueError(f"ndvi_map must be a 2D array, got {ndvi_map.ndim}D shape {ndvi_map.shape}")

    if prob_map.shape != ndvi_map.shape:
        raise ValueError(
            f"Shape mismatch: prob_map shape {prob_map.shape} != ndvi_map shape {ndvi_map.shape}"
        )

    if stress_map is not None:
        if not isinstance(stress_map, np.ndarray):
            raise TypeError("stress_map must be a numpy ndarray if provided.")
        if stress_map.ndim != 2:
            raise ValueError(f"stress_map must be a 2D array, got {stress_map.ndim}D shape {stress_map.shape}")
        if stress_map.shape != prob_map.shape:
            raise ValueError(
                f"Shape mismatch: stress_map shape {stress_map.shape} != prob_map shape {prob_map.shape}"
            )

    height, width = prob_map.shape
    records: List[Dict[str, Any]] = []

    for r in range(height):
        lat = origin_lat - r * pixel_spacing_deg
        for c in range(width):
            lon = origin_lon + c * pixel_spacing_deg

            prob_val = float(np.clip(prob_map[r, c], 0.0, 1.0))
            ndvi_val = float(ndvi_map[r, c])
            stress_val = float(stress_map[r, c]) if stress_map is not None else 0.0

            # Categorize risk and assign color palette
            if prob_val >= threshold:
                risk_label = "Infected"
                color_rgba = [235, 54, 54, 230]  # Vibrant alert red
            elif prob_val >= threshold * 0.5:
                risk_label = "Watch"
                color_rgba = [245, 166, 35, 200]  # Warning amber
            else:
                risk_label = "Healthy"
                color_rgba = [46, 184, 92, 180]   # Healthy vegetation green

            records.append({
                "row": int(r),
                "col": int(c),
                "lat": float(round(lat, 6)),
                "lon": float(round(lon, 6)),
                "probability": float(round(prob_val, 4)),
                "ndvi": float(round(ndvi_val, 4)),
                "stress": float(round(stress_val, 4)),
                "elevation": float(round(prob_val * 80.0, 2)),
                "risk_label": risk_label,
                "color_rgba": color_rgba,
            })

    return records


def simulate_temporal_spread(
    base_prob_map: np.ndarray,
    days: Optional[List[int]] = None,
    growth_rate: float = 0.06,
) -> Dict[int, np.ndarray]:
    """Model spatial blight progression over days using cellular diffusion and growth.

    Simulates epidemic fungal expansion across consecutive temporal horizons,
    enabling time-series animation and timeline scrubbing in the geospatial dashboard.

    Args:
        base_prob_map: Initial 2D disease probability raster (Day 0) of shape (H, W).
        days: List of target forecast days (default: [0, 3, 7, 14]).
        growth_rate: Logistic lesion proliferation and spore dispersal rate (default: 0.06).

    Returns:
        Dictionary mapping forecast day (int) to simulated 2D probability map (np.ndarray of shape (H, W)).

    Raises:
        ValueError: If base_prob_map is not 2D or growth_rate is non-positive.
    """
    if hasattr(base_prob_map, "detach"):
        base_prob_map = base_prob_map.detach().cpu().numpy()

    if not isinstance(base_prob_map, np.ndarray):
        raise TypeError("base_prob_map must be a numpy ndarray.")

    if base_prob_map.ndim != 2:
        raise ValueError(
            f"base_prob_map must be a 2D array, got {base_prob_map.ndim}D shape {base_prob_map.shape}"
        )

    if growth_rate <= 0.0:
        raise ValueError(f"growth_rate must be strictly positive, got {growth_rate}")

    target_days = sorted(list(set(days if days is not None else [0, 3, 7, 14])))
    if any(d < 0 for d in target_days):
        raise ValueError("Forecast days must be non-negative integers.")

    max_day = max(target_days) if target_days else 0

    current = np.clip(base_prob_map.copy(), 0.0, 1.0).astype(np.float32)
    results: Dict[int, np.ndarray] = {}

    if 0 in target_days:
        results[0] = current.copy()

    # Step day-by-day to simulate continuous spatial contagion
    for day_step in range(1, max_day + 1):
        # 3x3 local neighborhood diffusion with boundary padding
        padded = np.pad(current, 1, mode="edge")
        neighbor_mean = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            0.707 * (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
        ) / (4.0 + 4.0 * 0.707)

        # Logistic fungal proliferation
        growth = growth_rate * current * (1.0 - current + 0.1)

        # Spatial diffusion from infectious neighbors
        diffusion = (growth_rate * 0.5) * (neighbor_mean - current)

        current = current + growth + np.maximum(0.0, diffusion)
        current = np.clip(current, 0.0, 1.0).astype(np.float32)

        if day_step in target_days:
            results[day_step] = current.copy()

    # Return exactly the requested days in order
    return {d: results[d] for d in target_days}
