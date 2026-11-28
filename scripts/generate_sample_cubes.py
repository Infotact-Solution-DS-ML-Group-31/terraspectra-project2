"""Generate sample .npz hyperspectral datacubes for live Streamlit UI demos.

Creates:
    samples/healthy_field.npz    - Healthy vegetation canopy (minimal stress)
    samples/blight_outbreak.npz  - Significant pre-symptomatic fungal blight patch

Usage:
    python scripts/generate_sample_cubes.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

# Ensure repository root is importable when invoked directly as `python scripts/generate_sample_cubes.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.cube_loader import generate_synthetic_hyperspectral_cube


def generate_samples(output_dir: str | Path = "samples") -> tuple[Path, Path]:
    """Generate and save sample .npz hyperspectral cubes for demo uploads."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    healthy_cube, _ = generate_synthetic_hyperspectral_cube(
        height=32,
        width=32,
        num_bands=224,
        disease_zone_radius=1,
        seed=101,
    )
    healthy_path = out_path / "healthy_field.npz"
    np.savez_compressed(
        healthy_path,
        data=healthy_cube.data,
        wavelengths_nm=healthy_cube.wavelengths,
    )

    outbreak_cube, _ = generate_synthetic_hyperspectral_cube(
        height=32,
        width=32,
        num_bands=224,
        disease_zone_radius=10,
        seed=42,
    )
    outbreak_path = out_path / "blight_outbreak.npz"
    np.savez_compressed(
        outbreak_path,
        data=outbreak_cube.data,
        wavelengths_nm=outbreak_cube.wavelengths,
    )

    return healthy_path, outbreak_path


if __name__ == "__main__":
    p1, p2 = generate_samples()
    print(f"Generated sample cubes:\n  - {p1}\n  - {p2}")
