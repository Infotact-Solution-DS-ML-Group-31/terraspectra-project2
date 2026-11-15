"""Spectral index calculations and field health summarization for TerraSpectra.

Provides vegetation and biophysical indices computed from hyperspectral datacubes:
- Normalized Difference Vegetation Index (NDVI)
- Normalized Red-Edge Stress Index (NDRE / Red-Edge Stress)
- Field-level disease prevalence, area estimation, and health reporting.
"""

from __future__ import annotations

from typing import Any, Dict, Union
import numpy as np

from src.data.cube_loader import HyperspectralCube


def _get_cube_wavelengths(cube: HyperspectralCube) -> np.ndarray:
    """Safely extract spectral wavelengths array from a HyperspectralCube."""
    if hasattr(cube, "wavelengths_nm") and cube.wavelengths_nm is not None:
        return cube.wavelengths_nm
    elif hasattr(cube, "wavelengths") and cube.wavelengths is not None:
        return cube.wavelengths
    else:
        raise AttributeError(
            "HyperspectralCube must contain a 'wavelengths' or 'wavelengths_nm' array."
        )


def _find_nearest_band_idx(wavelengths: np.ndarray, target_nm: float) -> int:
    """Locate the index of the spectral band nearest to target_nm."""
    diffs = np.abs(wavelengths - target_nm)
    return int(np.argmin(diffs))


def compute_ndvi(
    cube: HyperspectralCube,
    red_nm: float = 670.0,
    nir_nm: float = 800.0,
) -> np.ndarray:
    """Compute the Normalized Difference Vegetation Index (NDVI) from a datacube.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red + 1e-8)

    Args:
        cube: HyperspectralCube containing spectral reflectance data and wavelengths.
        red_nm: Center wavelength for chlorophyll absorption in nanometers (default: 670.0 nm).
        nir_nm: Center wavelength for near-infrared canopy plateau in nanometers (default: 800.0 nm).

    Returns:
        2D numpy array of shape (height, width) with NDVI values bounded in [-1.0, 1.0].
    """
    if not isinstance(cube, HyperspectralCube):
        raise TypeError(f"Expected HyperspectralCube, got {type(cube)}")

    wavelengths = _get_cube_wavelengths(cube)

    red_idx = _find_nearest_band_idx(wavelengths, red_nm)
    nir_idx = _find_nearest_band_idx(wavelengths, nir_nm)

    red = cube.data[red_idx].astype(np.float32)
    nir = cube.data[nir_idx].astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_red_edge_stress(
    cube: HyperspectralCube,
    red_edge_nm: float = 710.0,
    nir_nm: float = 800.0,
) -> np.ndarray:
    """Compute Normalized Red-Edge Stress Index to highlight early fungal infection.

    Early fungal blight collapses the cellular red-edge reflectance transition
    prior to visible chlorosis or necrotic foliar breakdown.

    Formula:
        Stress = (NIR - RedEdge) / (NIR + RedEdge + 1e-8)

    Args:
        cube: HyperspectralCube containing spectral reflectance data and wavelengths.
        red_edge_nm: Wavelength in the red-edge transition in nanometers (default: 710.0 nm).
        nir_nm: Wavelength in the near-infrared plateau in nanometers (default: 800.0 nm).

    Returns:
        2D numpy array of shape (height, width) with values bounded in [-1.0, 1.0].
    """
    if not isinstance(cube, HyperspectralCube):
        raise TypeError(f"Expected HyperspectralCube, got {type(cube)}")

    wavelengths = _get_cube_wavelengths(cube)

    re_idx = _find_nearest_band_idx(wavelengths, red_edge_nm)
    nir_idx = _find_nearest_band_idx(wavelengths, nir_nm)

    red_edge = cube.data[re_idx].astype(np.float32)
    nir = cube.data[nir_idx].astype(np.float32)

    stress = (nir - red_edge) / (nir + red_edge + 1e-8)
    return np.clip(stress, -1.0, 1.0).astype(np.float32)


def summarize_field_health(
    prob_map: np.ndarray,
    ndvi_map: np.ndarray,
    threshold: float = 0.5,
    pixel_area_sq_m: float = 1.0,
) -> Dict[str, Any]:
    """Summarize geospatial crop health, disease prevalence, and infected acreage.

    Args:
        prob_map: 2D array of pixel-level fungal blight probability in [0.0, 1.0].
        ndvi_map: 2D array of NDVI vegetation index in [-1.0, 1.0].
        threshold: Decision threshold for classifying a pixel as infected (default: 0.5).
        pixel_area_sq_m: Ground surface area represented by one pixel in m^2 (default: 1.0).

    Returns:
        Dictionary containing:
            - 'total_pixels': Total number of evaluated pixels
            - 'infected_pixels': Count of pixels with probability >= threshold
            - 'infected_percentage': Percentage of total area diagnosed as infected
            - 'infected_area_hectares': Physical affected area in hectares (1 ha = 10,000 m^2)
            - 'mean_disease_probability': Field-wide average disease probability
            - 'mean_ndvi_healthy': Average NDVI of non-infected pixels
            - 'mean_ndvi_infected': Average NDVI of diagnosed infected pixels
            - 'severity_level': 'Low' (<10%), 'Moderate' (10%-25%), or 'High' (>25%)

    Raises:
        ValueError: If array dimensions are not 2D, shapes mismatch, or params are invalid.
    """
    # Accept torch tensors if passed
    if hasattr(prob_map, "detach"):
        prob_map = prob_map.detach().cpu().numpy()
    if hasattr(ndvi_map, "detach"):
        ndvi_map = ndvi_map.detach().cpu().numpy()

    if not isinstance(prob_map, np.ndarray) or not isinstance(ndvi_map, np.ndarray):
        raise TypeError("prob_map and ndvi_map must be numpy ndarrays or torch Tensors.")

    if prob_map.ndim != 2 or ndvi_map.ndim != 2:
        raise ValueError(
            f"prob_map and ndvi_map must be 2D arrays, got {prob_map.ndim}D and {ndvi_map.ndim}D"
        )

    if prob_map.shape != ndvi_map.shape:
        raise ValueError(
            f"Shape mismatch: prob_map shape {prob_map.shape} != ndvi_map shape {ndvi_map.shape}"
        )

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in range [0.0, 1.0], got {threshold}")

    if pixel_area_sq_m <= 0.0:
        raise ValueError(f"pixel_area_sq_m must be strictly positive, got {pixel_area_sq_m}")

    total_pixels = int(prob_map.size)
    infected_mask = prob_map >= threshold
    infected_pixels = int(np.sum(infected_mask))
    healthy_mask = ~infected_mask

    infected_percentage = float((infected_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0
    infected_area_hectares = float((infected_pixels * pixel_area_sq_m) / 10000.0)
    mean_disease_prob = float(np.mean(prob_map)) if total_pixels > 0 else 0.0

    mean_ndvi_healthy = (
        float(np.mean(ndvi_map[healthy_mask])) if np.any(healthy_mask) else 0.0
    )
    mean_ndvi_infected = (
        float(np.mean(ndvi_map[infected_mask])) if np.any(infected_mask) else 0.0
    )

    if infected_percentage < 10.0:
        severity_level = "Low"
    elif infected_percentage <= 25.0:
        severity_level = "Moderate"
    else:
        severity_level = "High"

    return {
        "total_pixels": total_pixels,
        "infected_pixels": infected_pixels,
        "infected_percentage": infected_percentage,
        "infected_area_hectares": infected_area_hectares,
        "mean_disease_probability": mean_disease_prob,
        "mean_ndvi_healthy": mean_ndvi_healthy,
        "mean_ndvi_infected": mean_ndvi_infected,
        "severity_level": severity_level,
    }
