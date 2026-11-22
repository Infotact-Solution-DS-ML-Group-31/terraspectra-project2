"""Architecture benchmarking utilities for TerraSpectra.

Provides lightweight, dependency-free profiling of forward-pass latency and
trainable parameter counts for the two TerraSpectra model architectures.

Public API
----------
count_trainable_parameters(model)
    Return the number of learnable parameters.

benchmark_model(model, model_name, input_tensor, target_mask, runs)
    Time forward passes and optionally compute segmentation metrics.

compare_architectures(height, width, pca_components, seed)
    End-to-end comparison of SpectralSpatial3DCNN vs HybridSpectralSpatialViT
    on a synthetic hyperspectral datacube.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.cube_loader import generate_synthetic_hyperspectral_cube
from src.data.pca_reducer import SpectralPCAReducer
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT
from src.models.metrics import evaluate_segmentation


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    """Return the total number of trainable (learnable) parameters in *model*.

    Args:
        model: Any ``torch.nn.Module`` instance.

    Returns:
        Integer count of parameters whose ``requires_grad`` flag is ``True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    runs: int = 3,
) -> Dict[str, Any]:
    """Profile a single forward pass through *model*.

    The function performs one warmup pass (not timed) followed by *runs*
    timed forward passes under ``torch.no_grad()`` and ``model.eval()``.
    Latency is measured with ``time.perf_counter()`` for maximum portability
    (no CUDA dependency required).

    Args:
        model: The ``nn.Module`` to benchmark.
        model_name: Human-readable label stored in the result dict.
        input_tensor: Batch input tensor — expected shape
            ``(1, 1, spectral_depth, H, W)`` for both architectures.
        target_mask: Optional ground-truth binary mask of shape ``(B, H, W)``
            or ``(H, W)``.  When provided, ``evaluate_segmentation`` is called
            on the final forward-pass logits and the results are stored under
            the ``"metrics"`` key.
        runs: Number of timed forward passes (default ``3``).

    Returns:
        Dictionary with the following keys:

        * ``"model_name"`` *(str)*
        * ``"trainable_params"`` *(int)*
        * ``"mean_latency_ms"`` *(float)*
        * ``"min_latency_ms"`` *(float)*
        * ``"max_latency_ms"`` *(float)*
        * ``"metrics"`` *(dict | None)* — present only when *target_mask* is
          not ``None``; contains keys ``iou``, ``dice``, ``precision``,
          ``recall``, ``pixel_accuracy``.

    Raises:
        ValueError: If *runs* is less than 1.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    model.eval()

    # --- Warmup (not timed) ------------------------------------------------
    with torch.no_grad():
        _ = model(input_tensor)

    # --- Timed runs ---------------------------------------------------------
    latencies_ms: List[float] = []
    final_logits: Optional[torch.Tensor] = None

    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            logits = model(input_tensor)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1_000.0)
        final_logits = logits  # keep last for metrics

    result: Dict[str, Any] = {
        "model_name": model_name,
        "trainable_params": count_trainable_parameters(model),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "metrics": None,
    }

    if target_mask is not None:
        # Ensure target_mask has the right shape for evaluate_segmentation
        if target_mask.ndim == 2:
            target_3d = target_mask.unsqueeze(0)  # (1, H, W)
        else:
            target_3d = target_mask

        result["metrics"] = evaluate_segmentation(
            logits=final_logits,
            targets=target_3d.long(),
        )

    return result


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def compare_architectures(
    height: int = 32,
    width: int = 32,
    pca_components: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Benchmark ``SpectralSpatial3DCNN`` and ``HybridSpectralSpatialViT`` side-by-side.

    Generates a synthetic 224-band hyperspectral cube, compresses it to
    *pca_components* spectral channels via ``SpectralPCAReducer``, constructs
    a matching input tensor, and runs :func:`benchmark_model` for both
    architectures.

    A synthetic binary target mask (derived from the simulated disease zone)
    is used to populate ``"metrics"`` in every result entry.

    Args:
        height: Spatial height of the synthetic raster in pixels (default ``32``).
        width: Spatial width of the synthetic raster in pixels (default ``32``).
        pca_components: Number of PCA spectral components (default ``15``).
        seed: Random seed for reproducible cube generation (default ``42``).

    Returns:
        A list of two benchmark dicts (``"cnn3d"`` first, then ``"hybrid"``)
        in the format produced by :func:`benchmark_model`.
    """
    # 1 — Synthetic hyperspectral cube
    cube, disease_mask = generate_synthetic_hyperspectral_cube(
        height=height,
        width=width,
        num_bands=224,
        disease_zone_radius=max(4, min(height, width) // 8),
        seed=seed,
    )

    # 2 — PCA dimensionality reduction  (C, H, W) with C = pca_components
    reducer = SpectralPCAReducer(n_components=pca_components, random_state=seed)
    reduced: np.ndarray = reducer.fit_transform(cube)  # (pca_components, H, W)

    # 3 — Build input tensor  (1, 1, pca_components, H, W)
    input_tensor = torch.from_numpy(reduced).float().unsqueeze(0).unsqueeze(0)

    # 4 — Binary ground-truth mask  (1, H, W)
    target_mask = torch.from_numpy(disease_mask.astype(np.int64)).unsqueeze(0)

    # 5 — Instantiate models in eval mode
    cnn_model = SpectralSpatial3DCNN(in_channels=1, num_classes=2)
    cnn_model.eval()

    hybrid_model = HybridSpectralSpatialViT(
        in_channels=1,
        cnn_features=32,
        vit_embed_dim=64,
        patch_size=4,
        num_classes=2,
    )
    hybrid_model.eval()

    # 6 — Benchmark both architectures
    results = [
        benchmark_model(
            model=cnn_model,
            model_name="cnn3d",
            input_tensor=input_tensor,
            target_mask=target_mask,
            runs=3,
        ),
        benchmark_model(
            model=hybrid_model,
            model_name="hybrid",
            input_tensor=input_tensor,
            target_mask=target_mask,
            runs=3,
        ),
    ]

    return results
