"""Model checkpoint serialisation and deserialisation for TerraSpectra.

Provides two public helpers:

* ``save_model_checkpoint`` — persists a model's state dict together with
  lightweight metadata to a ``.pt`` file.
* ``load_model_checkpoint`` — reconstructs the original model architecture
  from the saved file, loads weights, and returns the model in eval mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def save_model_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    model_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a TerraSpectra model checkpoint to disk.

    The checkpoint file is a plain ``torch.save`` dictionary with the
    following keys:

    * ``"model_type"`` — architecture identifier (``"hybrid"`` or ``"cnn3d"``).
    * ``"state_dict"`` — the model's ``state_dict()``.
    * ``"metadata"``   — an arbitrary user-supplied dictionary (may be ``{}``).

    Args:
        model: The ``nn.Module`` instance to checkpoint.
        path: Destination file path (a ``.pt`` or ``.pth`` extension is
            recommended but not enforced).
        model_type: Architecture identifier — ``"hybrid"`` or ``"cnn3d"``.
        metadata: Optional dictionary of extra information to embed (e.g.,
            training epoch, loss history, PCA components used).

    Returns:
        The resolved ``Path`` object of the written checkpoint file.

    Raises:
        ValueError: If ``model_type`` is not ``"hybrid"`` or ``"cnn3d"``.
    """
    _validate_model_type(model_type)

    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "metadata": metadata if metadata is not None else {},
    }
    torch.save(payload, dest)
    return dest


def load_model_checkpoint(
    path: Union[str, Path],
    in_channels: int = 15,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a TerraSpectra model checkpoint from disk.

    Instantiates the correct model architecture based on the ``model_type``
    stored inside the checkpoint, loads the saved weights, and returns the
    model in evaluation mode.

    Args:
        path: Path to the ``.pt`` checkpoint file written by
            :func:`save_model_checkpoint`.
        in_channels: Spectral depth fed into the model.  For the default
            pipeline this is the number of PCA components (default ``15``).
            The value must match what was used during training/saving.
        device: Torch device string for weight loading (default ``"cpu"``).

    Returns:
        A 2-tuple ``(model, metadata)`` where *model* is the reconstructed
        ``nn.Module`` in eval mode and *metadata* is the dictionary that was
        embedded at save time (may be ``{}``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``model_type`` stored in the checkpoint is unrecognised.
    """
    src_path = Path(path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: '{src_path}'"
        )

    payload: Dict[str, Any] = torch.load(src_path, map_location=device, weights_only=False)

    model_type: str = payload.get("model_type", "")
    _validate_model_type(model_type)

    model = _build_model(model_type, in_channels=in_channels)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    metadata: Dict[str, Any] = payload.get("metadata", {})
    return model, metadata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_SUPPORTED_MODEL_TYPES = frozenset({"hybrid", "cnn3d"})


def _validate_model_type(model_type: str) -> None:
    """Raise ``ValueError`` for unknown architecture identifiers."""
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported values: {sorted(_SUPPORTED_MODEL_TYPES)}"
        )


def _build_model(model_type: str, in_channels: int) -> nn.Module:
    """Instantiate a fresh (un-trained) model of the requested architecture."""
    if model_type == "hybrid":
        return HybridSpectralSpatialViT(
            in_channels=1,
            cnn_features=32,
            vit_embed_dim=64,
            patch_size=4,
            num_classes=2,
        )
    # model_type == "cnn3d"
    return SpectralSpatial3DCNN(
        in_channels=1,
        num_classes=2,
    )
