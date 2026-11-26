"""FastAPI application entrypoint for TerraSpectra.

Exposes REST microservices for model health verification and full-field
hyperspectral crop disease forecasting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from fastapi import FastAPI, HTTPException, Query
import torch.nn as nn

from src.analysis.spectral_index import (
    compute_ndvi,
    compute_red_edge_stress,
    summarize_field_health,
)
from src.api.schemas import (
    BenchmarkResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from src.api.tiling import TilingConfig, predict_full_raster
from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer
from src.models.benchmark import compare_architectures
from src.models.checkpoint import load_model_checkpoint
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT

app = FastAPI(
    title="TerraSpectra API",
    version="1.0.0",
    description="Hyperspectral Crop Disease Forecasting and Early Fungal Blight Detection.",
)

# In-memory model registry for fast inference
_MODEL_REGISTRY: Dict[str, nn.Module] = {
    "hybrid": HybridSpectralSpatialViT(
        in_channels=1,
        cnn_features=32,
        vit_embed_dim=64,
        patch_size=4,
        num_classes=2,
    ),
    "cnn3d": SpectralSpatial3DCNN(
        in_channels=1,
        num_classes=2,
    ),
}

# Set all models to evaluation mode
for model in _MODEL_REGISTRY.values():
    model.eval()


def get_model(model_type: str) -> nn.Module:
    """Retrieve or instantiate requested neural model."""
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Available models: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_type]


def get_model_for_inference(
    model_type: str,
    pca_components: int,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Tuple[nn.Module, bool]:
    """Retrieve a model for inference, loading trained demo weights if available.

    Checks whether ``<checkpoint_dir>/<model_type>_demo.pt`` exists and
    ``pca_components == 15``. If so, loads the trained weights via
    :func:`load_model_checkpoint` and returns ``(model, True)``. Otherwise,
    instantiates a fresh model in evaluation mode and returns ``(model, False)``.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    ckpt_path = Path(checkpoint_dir) / f"{model_type}_demo.pt"
    if pca_components == 15 and ckpt_path.is_file():
        loaded_model, _ = load_model_checkpoint(
            ckpt_path,
            in_channels=pca_components,
            device="cpu",
        )
        return loaded_model, True

    if model_type == "hybrid":
        fresh_model: nn.Module = HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
    else:
        fresh_model = SpectralSpatial3DCNN(
            in_channels=1,
            num_classes=2,
        )
    fresh_model.eval()
    return fresh_model, False


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint confirming API service and neural model readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=True,
        available_models=list(_MODEL_REGISTRY.keys()),
    )


_DEFAULT_PCA_COMPONENTS = 15
_CHECKPOINT_DIR = Path("checkpoints")


@app.get("/api/v1/models")
def list_models() -> Dict[str, Any]:
    """Return available model architectures and configuration metadata.

    Returns:
        JSON object with:
        - ``architectures``: List of supported model type identifiers.
        - ``default_pca_components``: Default spectral PCA components used
          for inference (must match training configuration).
        - ``checkpoint_dir_exists``: Whether the local ``checkpoints/``
          directory exists on this server instance.
        - ``checkpoint_dir``: Relative path to the checkpoint directory.
    """
    return {
        "architectures": list(_MODEL_REGISTRY.keys()),
        "default_pca_components": _DEFAULT_PCA_COMPONENTS,
        "checkpoint_dir_exists": _CHECKPOINT_DIR.is_dir(),
        "checkpoint_dir": str(_CHECKPOINT_DIR),
    }


@app.post("/api/v1/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    """Run full-field hyperspectral crop disease forecasting and biophysical analysis."""
    try:
        seed = request.seed if request.seed is not None else 42

        # 1. Simulate continuous imaging spectrometer datacube (400-2500nm)
        cube, _ = generate_synthetic_hyperspectral_cube(
            height=request.height,
            width=request.width,
            num_bands=request.bands,
            seed=seed,
        )

        # 2. Compute biophysical indices
        ndvi_map = compute_ndvi(cube)
        stress_map = compute_red_edge_stress(cube)

        # 3. Spectral dimensionality reduction
        reducer = SpectralPCAReducer(n_components=request.pca_components, random_state=seed)
        reduced_raster = reducer.fit_transform(cube)

        # 4. Neural inference via overlapping spatial tiling
        model, _ = get_model_for_inference(
            model_type=request.model_type,
            pca_components=request.pca_components,
            checkpoint_dir=_CHECKPOINT_DIR,
        )
        tiling_cfg = TilingConfig(patch_size=16, stride=8, batch_size=8, device="cpu")
        prob_map = predict_full_raster(model, reduced_raster, config=tiling_cfg)

        # 5. Geospatial field health summarization
        health_summary = summarize_field_health(
            prob_map=prob_map,
            ndvi_map=ndvi_map,
            threshold=request.disease_threshold,
            pixel_area_sq_m=request.pixel_area_sq_m,
        )

        return ForecastResponse(
            model_type=request.model_type,
            raster_shape=[request.height, request.width],
            health_summary=health_summary,
            probability_map=prob_map.tolist(),
            ndvi_map=ndvi_map.tolist(),
            red_edge_stress_map=stress_map.tolist(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/v1/benchmark", response_model=BenchmarkResponse)
def get_benchmark(
    height: int = Query(default=32, ge=16, le=128, description="Spatial height in pixels."),
    width: int = Query(default=32, ge=16, le=128, description="Spatial width in pixels."),
    pca_components: int = Query(default=15, ge=3, le=64, description="Target PCA spectral components."),
    seed: int = Query(default=42, description="Random seed for sensor simulation."),
) -> BenchmarkResponse:
    """Run side-by-side architecture benchmark profiling 3D-CNN vs Hybrid ViT."""
    try:
        results = compare_architectures(
            height=height,
            width=width,
            pca_components=pca_components,
            seed=seed,
        )
        return BenchmarkResponse(
            height=height,
            width=width,
            pca_components=pca_components,
            results=results,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")

