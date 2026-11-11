"""Hyperspectral Cube Loader and Synthetic Simulator for TerraSpectra.

Provides data structures and synthetic generation utilities for simulating
airborne/satellite imaging spectrometer cubes (e.g., NASA Hyperion, ESA CHIME,
PRISMA, EnMAP, Sentinel-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class HyperspectralCube:
    """Hyperspectral datacube container.

    Attributes:
        data: Spectral reflectance array of shape (num_bands, height, width)
              normalized to [0.0, 1.0].
        wavelengths: Continuous spectral band centers in nanometers (shape: (num_bands,)).
        spatial_resolution_meters: Ground sampling distance in meters (e.g., 3.0 m/px).
        metadata: Contextual mission, sensor, and acquisition metadata.
    """

    data: np.ndarray
    wavelengths: np.ndarray
    spatial_resolution_meters: float = 3.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor shapes and value domain."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be a numpy ndarray, got {type(self.data)}")
        if self.data.ndim != 3:
            raise ValueError(
                f"data must have shape (num_bands, height, width), got {self.data.shape}"
            )
        if not isinstance(self.wavelengths, np.ndarray):
            raise TypeError(
                f"wavelengths must be a numpy ndarray, got {type(self.wavelengths)}"
            )
        if self.wavelengths.ndim != 1:
            raise ValueError(
                f"wavelengths must be 1-dimensional, got shape {self.wavelengths.shape}"
            )
        if self.data.shape[0] != len(self.wavelengths):
            raise ValueError(
                f"Band count mismatch: data has {self.data.shape[0]} bands, "
                f"but wavelengths has {len(self.wavelengths)} elements."
            )
        if self.spatial_resolution_meters <= 0:
            raise ValueError(
                f"spatial_resolution_meters must be positive, got {self.spatial_resolution_meters}"
            )

    @property
    def num_bands(self) -> int:
        """Number of spectral channels."""
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        """Spatial height in pixels."""
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Spatial width in pixels."""
        return int(self.data.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Tensor dimensions (num_bands, height, width)."""
        return self.data.shape


def _generate_healthy_canopy_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Model healthy vegetation spectral reflectance profile.

    Replicates real-world photosynthetic canopy signatures:
    - Visible absorption (400-700 nm): low reflectance with blue/red chlorophyll
      absorption dips (~450 nm & ~680 nm) and green reflectance peak (~550 nm).
    - Red-edge transition (700-750 nm): steep reflectance increase.
    - Near-Infrared plateau (750-1100 nm): high cell-structure scattering plateau (~0.50-0.55).
    - Short-Wave Infrared (1100-2500 nm): atmospheric & foliar water absorption
      troughs (~1400 nm and ~1900 nm).
    """
    w = wavelengths

    # Visible spectrum baseline (400 - 700 nm)
    vis_base = 0.05
    green_peak = 0.07 * np.exp(-0.5 * ((w - 550.0) / 35.0) ** 2)
    chl_dip = 0.025 * np.exp(-0.5 * ((w - 675.0) / 25.0) ** 2)
    vis = vis_base + green_peak - chl_dip

    # Red-edge sigmoid transition (700 - 750 nm)
    red_edge = 1.0 / (1.0 + np.exp(-(w - 715.0) / 10.0))

    # NIR plateau (750 - 1100 nm) with gentle slope
    nir = 0.52 - 0.00007 * np.maximum(0.0, w - 800.0)

    # Primary blend
    spectrum = (1.0 - red_edge) * vis + red_edge * nir

    # Water absorption bands at 1400 nm and 1900 nm
    water_1400 = 0.25 * np.exp(-0.5 * ((w - 1400.0) / 45.0) ** 2)
    water_1900 = 0.28 * np.exp(-0.5 * ((w - 1920.0) / 50.0) ** 2)
    swir_drop = 0.12 * np.maximum(0.0, (w - 1800.0) / 700.0)

    spectrum = spectrum - water_1400 - water_1900 - swir_drop
    return np.clip(spectrum, 0.02, 0.98)


def generate_synthetic_hyperspectral_cube(
    height: int = 64,
    width: int = 64,
    num_bands: int = 224,
    disease_zone_radius: int = 8,
    seed: int = 42,
) -> Tuple[HyperspectralCube, np.ndarray]:
    """Generate a synthetic hyperspectral datacube with an early fungal blight infection zone.

    Simulates continuous hyperspectral imagery from 400nm to 2500nm, mimicking
    NASA Hyperion / ESA Sentinel sensors.

    Models early-stage fungal blight where chlorophyll absorption and cell structure
    breakdown drop the red-edge reflectance by 15-25% without causing visible
    leaf chlorosis or necrosis in the visible spectrum.

    Args:
        height: Spatial height in pixels.
        width: Spatial width in pixels.
        num_bands: Number of contiguous spectral bands (default 224).
        disease_zone_radius: Radius in pixels of circular disease outbreak zone.
        seed: Random seed for reproducible spatial and sensor noise.

    Returns:
        Tuple containing:
        - HyperspectralCube: Datacube with shape (num_bands, height, width).
        - np.ndarray: Binary disease mask of shape (height, width).
    """
    rng = np.random.default_rng(seed)

    # 1. Generate 224 continuous spectral wavelengths (400nm to 2500nm)
    wavelengths = np.linspace(400.0, 2500.0, num_bands, dtype=np.float32)

    # 2. Base healthy vegetation spectral curve
    base_spectrum = _generate_healthy_canopy_spectrum(wavelengths)  # shape: (num_bands,)

    # 3. Spatial field variability (micro-topography, canopy density)
    spatial_factor = 1.0 + rng.normal(loc=0.0, scale=0.03, size=(height, width))
    spatial_factor = np.clip(spatial_factor, 0.90, 1.10)

    # Broadcast to 3D cube (num_bands, height, width)
    cube_data = base_spectrum[:, np.newaxis, np.newaxis] * spatial_factor[np.newaxis, :, :]

    # 4. Inject circular disease outbreak zone (5-acre patch equivalent)
    center_y, center_x = height // 2, width // 2
    yy, xx = np.ogrid[:height, :width]
    dist_sq = (yy - center_y) ** 2 + (xx - center_x) ** 2
    disease_mask = (dist_sq <= disease_zone_radius ** 2).astype(np.uint8)

    # Identify red-edge spectrum (700nm - 750nm)
    red_edge_idx = np.where((wavelengths >= 700.0) & (wavelengths <= 750.0))[0]

    if len(red_edge_idx) > 0 and np.any(disease_mask):
        # Calculate distance-dependent infection severity (15% to 25% drop)
        dist = np.sqrt(dist_sq)
        norm_dist = np.clip(dist / max(disease_zone_radius, 1), 0.0, 1.0)
        # Core infection is ~25% drop, boundary is ~15% drop
        infection_drop = 0.25 - 0.10 * norm_dist  # range [0.15, 0.25]
        retention = 1.0 - infection_drop

        # Apply drop specifically to red-edge bands for infected pixels
        # Visible bands (<700nm) remain unperturbed, modeling pre-symptomatic phase
        for b in red_edge_idx:
            cube_data[b, disease_mask == 1] *= retention[disease_mask == 1]

    # 5. Add small sensor noise (shot noise / dark current with SNR ~ 500:1)
    noise = rng.normal(loc=0.0, scale=0.001, size=cube_data.shape)
    cube_data = np.clip(cube_data + noise, 0.0, 1.0).astype(np.float32)

    metadata = {
        "sensor": "NASA-Hyperion-Simulated",
        "spectral_range_nm": (float(wavelengths[0]), float(wavelengths[-1])),
        "num_bands": num_bands,
        "disease_type": "Early Fungal Blight (pre-symptomatic)",
        "disease_zone_radius_px": disease_zone_radius,
        "disease_zone_center": (center_y, center_x),
        "spatial_resolution_meters": 3.0,
        "estimated_affected_acres": 5.0,
    }

    cube = HyperspectralCube(
        data=cube_data,
        wavelengths=wavelengths,
        spatial_resolution_meters=3.0,
        metadata=metadata,
    )

    return cube, disease_mask

