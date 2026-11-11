"""Spectral PCA Dimensionality Reducer for Hyperspectral Datacubes.

Reduces high-dimensional continuous hyperspectral bands into lower-dimensional
principal spectral representations while preserving spatial structure and
maximum spectral variance.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA

from src.data.cube_loader import HyperspectralCube

logger = logging.getLogger(__name__)


class SpectralPCAReducer:
    """Principal Component Analysis reducer for hyperspectral datacubes.

    Compresses raw cubes of shape (num_bands, height, width) to
    (n_components, height, width), fitting PCA on spectral vectors across
    all spatial pixels (height * width, num_bands).
    """

    def __init__(self, n_components: int = 15, random_state: Optional[int] = 42) -> None:
        """Initialize SpectralPCAReducer.

        Args:
            n_components: Target number of principal spectral components.
            random_state: Seed for random state reproducibility.
        """
        if n_components <= 0:
            raise ValueError(f"n_components must be positive, got {n_components}")

        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False
        self.num_bands_in_: Optional[int] = None
        self._explained_variance_ratio_: Optional[np.ndarray] = None

    @property
    def cumulative_explained_variance_ratio(self) -> float:
        """Cumulative explained variance ratio of fitted PCA components."""
        if not self.is_fitted or self._explained_variance_ratio_ is None:
            raise ValueError("SpectralPCAReducer must be fitted before querying variance.")
        return float(np.sum(self._explained_variance_ratio_))

    def verify_cumulative_variance(self, threshold: float = 0.95) -> bool:
        """Check whether cumulative explained variance ratio meets or exceeds threshold.

        Args:
            threshold: Minimum required cumulative variance ratio (default: 0.95).

        Returns:
            True if cumulative variance >= threshold, False otherwise.
        """
        return self.cumulative_explained_variance_ratio >= threshold

    def _extract_and_validate_data(
        self, cube: Union[HyperspectralCube, np.ndarray]
    ) -> Tuple[np.ndarray, int, int, int]:
        """Extract array data and validate 3D tensor shape.

        Returns:
            Tuple of (data_array, num_bands, height, width).
        """
        if isinstance(cube, HyperspectralCube):
            data = cube.data
        elif isinstance(cube, np.ndarray):
            data = cube
        else:
            raise TypeError(
                f"Expected HyperspectralCube or np.ndarray, got {type(cube)}"
            )

        if data.ndim != 3:
            raise ValueError(
                f"Input tensor must have shape (num_bands, height, width), got {data.shape}"
            )

        num_bands, height, width = data.shape
        return data, num_bands, height, width

    def fit(self, cube: Union[HyperspectralCube, np.ndarray]) -> "SpectralPCAReducer":
        """Fit PCA model on spatial pixels of the hyperspectral cube.

        Reshapes spatial pixels from (num_bands, height, width) to
        (height * width, num_bands).

        Args:
            cube: Input HyperspectralCube or numpy array of shape (num_bands, height, width).

        Returns:
            self
        """
        data, num_bands, height, width = self._extract_and_validate_data(cube)

        if self.n_components > num_bands:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed num_bands ({num_bands})"
            )

        # Transpose (num_bands, height, width) -> (height, width, num_bands)
        # Reshape to (height * width, num_bands)
        pixels = np.transpose(data, (1, 2, 0)).reshape(-1, num_bands)

        self.pca.fit(pixels)
        self.num_bands_in_ = num_bands
        self._explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.is_fitted = True

        cum_var = self.cumulative_explained_variance_ratio
        if cum_var < 0.95:
            logger.warning(
                "Cumulative explained variance ratio (%.4f) is below the 0.95 threshold.",
                cum_var,
            )

        return self

    def transform(self, cube: Union[HyperspectralCube, np.ndarray]) -> np.ndarray:
        """Project hyperspectral cube down to principal spectral components.

        Args:
            cube: Input HyperspectralCube or numpy array of shape (num_bands, height, width).

        Returns:
            Compressed array of shape (n_components, height, width) as float32.
        """
        if not self.is_fitted:
            raise RuntimeError("SpectralPCAReducer is not fitted yet. Call .fit() first.")

        data, num_bands, height, width = self._extract_and_validate_data(cube)

        if num_bands != self.num_bands_in_:
            raise ValueError(
                f"Feature dimension mismatch: fitted on {self.num_bands_in_} bands, "
                f"got {num_bands} bands."
            )

        # Reshape to (height * width, num_bands)
        pixels = np.transpose(data, (1, 2, 0)).reshape(-1, num_bands)

        # PCA transform -> (height * width, n_components)
        reduced_pixels = self.pca.transform(pixels)

        # Reshape back to (height, width, n_components) then transpose to (n_components, height, width)
        reduced_tensor = reduced_pixels.reshape(height, width, self.n_components)
        reduced_tensor = np.transpose(reduced_tensor, (2, 0, 1))

        return reduced_tensor.astype(np.float32)

    def fit_transform(self, cube: Union[HyperspectralCube, np.ndarray]) -> np.ndarray:
        """Fit PCA and transform cube in a single pass.

        Args:
            cube: Input HyperspectralCube or numpy array of shape (num_bands, height, width).

        Returns:
            Compressed array of shape (n_components, height, width).
        """
        return self.fit(cube).transform(cube)

    def inverse_transform(self, reduced_tensor: np.ndarray) -> np.ndarray:
        """Reconstruct original spectral band dimensions from principal components.

        Args:
            reduced_tensor: Compressed array of shape (n_components, height, width).

        Returns:
            Reconstructed array of shape (num_bands, height, width) as float32.
        """
        if not self.is_fitted:
            raise RuntimeError("SpectralPCAReducer is not fitted yet. Call .fit() first.")

        if not isinstance(reduced_tensor, np.ndarray):
            raise TypeError(
                f"reduced_tensor must be np.ndarray, got {type(reduced_tensor)}"
            )

        if reduced_tensor.ndim != 3:
            raise ValueError(
                f"reduced_tensor must have shape (n_components, height, width), "
                f"got {reduced_tensor.shape}"
            )

        n_comp, height, width = reduced_tensor.shape
        if n_comp != self.n_components:
            raise ValueError(
                f"Dimension mismatch: expected {self.n_components} components, got {n_comp}"
            )

        # Reshape to (height * width, n_components)
        reduced_pixels = np.transpose(reduced_tensor, (1, 2, 0)).reshape(-1, n_comp)

        # PCA inverse_transform -> (height * width, num_bands_in_)
        reconstructed_pixels = self.pca.inverse_transform(reduced_pixels)

        # Reshape back to (height, width, num_bands_in_) then transpose to (num_bands, height, width)
        reconstructed_tensor = reconstructed_pixels.reshape(
            height, width, self.num_bands_in_
        )
        reconstructed_tensor = np.transpose(reconstructed_tensor, (2, 0, 1))

        return reconstructed_tensor.astype(np.float32)

