"""TerraSpectra Command-Line Interface.

Runs the full hyperspectral crop disease forecasting pipeline and prints a
clean field health summary to stdout.  Optionally writes the JSON summary to a
file.

Usage::

    python -m src.cli --help
    python -m src.cli --height 64 --width 64 --model hybrid --threshold 0.5
    python -m src.cli --model cnn3d --output summary.json

The module is also importable so that tests can call ``main(argv=[...])``
directly without spawning a subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

from src.analysis.spectral_index import (
    compute_ndvi,
    compute_red_edge_stress,
    summarize_field_health,
)
from src.api.tiling import TilingConfig, predict_full_raster
from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "TerraSpectra: Hyperspectral Crop Disease Forecasting CLI.\n"
            "Generates a synthetic hyperspectral cube, runs PCA compression,\n"
            "neural segmentation inference, and prints a field health summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        metavar="PIXELS",
        help="Raster height in pixels (default: 64).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        metavar="PIXELS",
        help="Raster width in pixels (default: 64).",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=224,
        metavar="N",
        help="Number of spectral bands to simulate (default: 224).",
    )
    parser.add_argument(
        "--pca-components",
        dest="pca_components",
        type=int,
        default=15,
        metavar="K",
        help="PCA components for spectral dimensionality reduction (default: 15).",
    )
    parser.add_argument(
        "--model",
        choices=["hybrid", "cnn3d"],
        default="hybrid",
        help="Neural architecture: 'hybrid' (3D-CNN + ViT) or 'cnn3d' (default: hybrid).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="PROB",
        help="Blight classification probability threshold [0.0, 1.0] (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducible synthetic cube generation (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional path to write the JSON health summary (e.g. summary.json).",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_model(model_type: str):
    """Instantiate and return the requested neural model in eval mode."""
    if model_type == "hybrid":
        model = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
    elif model_type == "cnn3d":
        model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")
    model.eval()
    return model


def _run_pipeline(
    height: int,
    width: int,
    num_bands: int,
    pca_components: int,
    model_type: str,
    threshold: float,
    seed: int,
) -> dict:
    """Execute the full TerraSpectra inference pipeline.

    Returns:
        dict: Field health summary produced by ``summarize_field_health``.
    """
    # 1 — Simulate hyperspectral datacube (bands x height x width)
    cube, _ = generate_synthetic_hyperspectral_cube(
        height=height,
        width=width,
        num_bands=num_bands,
        disease_zone_radius=max(4, min(height, width) // 8),
        seed=seed,
    )

    # 2 — Biophysical spectral indices
    ndvi_map = compute_ndvi(cube)
    _stress_map = compute_red_edge_stress(cube)  # noqa: F841 (unused in summary)

    # 3 — Spectral PCA dimensionality reduction
    reducer = SpectralPCAReducer(n_components=pca_components, random_state=seed)
    reduced_raster = reducer.fit_transform(cube)

    # 4 — Overlapping spatial tiling inference
    model = _build_model(model_type)
    tiling_cfg = TilingConfig(patch_size=16, stride=8, batch_size=8, device="cpu")
    prob_map = predict_full_raster(model, reduced_raster, config=tiling_cfg)

    # 5 — Field health summarization
    health_summary = summarize_field_health(
        prob_map=prob_map,
        ndvi_map=ndvi_map,
        threshold=threshold,
        pixel_area_sq_m=1.0,
    )

    return health_summary


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_summary(summary: dict, model_type: str) -> str:
    """Render a clean, human-readable summary block."""
    lines = [
        "",
        "┌─────────────────────────────────────────────────────┐",
        "│       TerraSpectra — Field Health Report             │",
        "└─────────────────────────────────────────────────────┘",
        f"  Model Architecture  : {model_type}",
        f"  Severity Level      : {summary.get('severity_level', 'N/A')}",
        f"  Total Pixels        : {summary.get('total_pixels', 0)}",
        f"  Infected Pixels     : {summary.get('infected_pixels', 0)}",
        f"  Infected Area       : {summary.get('infected_percentage', 0.0):.2f}%",
        f"  Infected Hectares   : {summary.get('infected_area_hectares', 0.0):.4f} ha",
        f"  Mean Disease Prob.  : {summary.get('mean_disease_probability', 0.0):.4f}",
        f"  Mean NDVI (Healthy) : {summary.get('mean_ndvi_healthy', float('nan')):.4f}",
        f"  Mean NDVI (Infected): {summary.get('mean_ndvi_infected', float('nan')):.4f}",
        "─" * 55,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (uses ``sys.argv[1:]`` when ``None``).

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Validate argument ranges ------------------------------------------
    errors: List[str] = []
    if args.height < 16:
        errors.append("--height must be >= 16.")
    if args.width < 16:
        errors.append("--width must be >= 16.")
    if args.bands < 15:
        errors.append("--bands must be >= 15.")
    if args.pca_components < 3 or args.pca_components > args.bands:
        errors.append(
            f"--pca-components must be in [3, --bands={args.bands}]."
        )
    if not (0.0 <= args.threshold <= 1.0):
        errors.append("--threshold must be in [0.0, 1.0].")
    if errors:
        for msg in errors:
            print(f"[error] {msg}", file=sys.stderr)
        return 1

    # --- Run pipeline -------------------------------------------------------
    try:
        summary = _run_pipeline(
            height=args.height,
            width=args.width,
            num_bands=args.bands,
            pca_components=args.pca_components,
            model_type=args.model,
            threshold=args.threshold,
            seed=args.seed,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[error] Pipeline failed: {exc}", file=sys.stderr)
        return 1

    # --- Print human-readable summary ---------------------------------------
    print(_format_summary(summary, model_type=args.model))

    # --- Optionally write JSON output ---------------------------------------
    if args.output:
        # Convert any numpy scalars to plain Python types for JSON serialisation
        serialisable: dict = {}
        for k, v in summary.items():
            if isinstance(v, (np.floating, np.integer)):
                serialisable[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                serialisable[k] = v

        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(serialisable, fh, indent=2)
            print(f"[info] JSON summary written to: {args.output}")
        except OSError as exc:  # pragma: no cover
            print(f"[error] Could not write output file: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
