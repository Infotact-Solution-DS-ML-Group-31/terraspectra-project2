"""Pydantic v2 schemas for TerraSpectra API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    available_models: List[str]


class ArchitectureBenchmarkItem(BaseModel):
    """Benchmark result for a single model architecture."""

    model_name: str
    trainable_params: int
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    metrics: Optional[Dict[str, float]] = None


class BenchmarkResponse(BaseModel):
    """Side-by-side architecture benchmark response schema."""

    height: int
    width: int
    pca_components: int
    results: List[ArchitectureBenchmarkItem]


class ForecastRequest(BaseModel):
    """Hyperspectral crop disease forecast request schema."""

    height: int = Field(default=64, ge=16, le=256, description="Spatial height in pixels.")
    width: int = Field(default=64, ge=16, le=256, description="Spatial width in pixels.")
    bands: int = Field(default=224, ge=15, le=300, description="Number of continuous spectral bands.")
    pca_components: int = Field(default=15, ge=3, le=64, description="Target PCA spectral components.")
    model_type: Literal["hybrid", "cnn3d"] = Field(
        default="hybrid", description="Neural architecture for disease segmentation."
    )
    disease_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Binary classification probability threshold."
    )
    pixel_area_sq_m: float = Field(
        default=1.0, gt=0.0, description="Ground sampling area per pixel in square meters."
    )
    seed: Optional[int] = Field(default=42, description="Random seed for reproducible sensor simulation.")

    @model_validator(mode="after")
    def validate_components(self) -> "ForecastRequest":
        """Ensure PCA components do not exceed simulated input spectral bands."""
        if self.pca_components > self.bands:
            raise ValueError(
                f"pca_components ({self.pca_components}) cannot exceed bands ({self.bands})"
            )
        return self


class ForecastResponse(BaseModel):
    """Hyperspectral crop disease forecast response schema."""

    model_type: str
    raster_shape: List[int]
    health_summary: Dict[str, Any]
    probability_map: List[List[float]]
    ndvi_map: List[List[float]]
    red_edge_stress_map: List[List[float]]
