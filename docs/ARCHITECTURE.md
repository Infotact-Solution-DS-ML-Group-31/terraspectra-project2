# TerraSpectra Architecture

TerraSpectra is a **100% Pure-Python** geospatial AI system for early hyperspectral crop disease (fungal blight) detection and multi-day spread forecasting. It simulates continuous 224-band imaging spectrometer datacubes (400–2500 nm), compresses spectral redundancy via Principal Component Analysis, performs 3D-CNN + Vision Transformer segmentation, and renders interactive 3D WebGL risk maps through Streamlit and PyDeck.

```
[Hyperspectral Cube (224xHxW)]
        │
        ├──► [Biophysical Indices] ──► NDVI & Red-Edge Stress Maps
        │
        └──► [Spectral PCA Reducer] ──► (15xHxW)
                    │
                    ▼
          [3D-CNN / Hybrid ViT] ──► (Sliding-Window Tiling)
                    │
                    ▼
          [Probability Map (HxW)]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[Field Health Summary]   [Geogrid & Temporal Spread]
        │                       │
        └───────────┬───────────┘
                    ▼
    [FastAPI REST API / Streamlit + PyDeck 3D GIS / CLI]
```

---

## 1. Data & PCA Compression (`src/data/`)

- **`src/data/cube_loader.py`**:
  - Defines the immutable `HyperspectralCube` frozen dataclass (`data` of shape `(num_bands, height, width)` normalized to `[0.0, 1.0]`, `wavelengths` spanning 400–2500 nm, `spatial_resolution_meters`, and `metadata`).
  - `generate_synthetic_hyperspectral_cube(...)`: Simulates realistic vegetation reflectance spectra with chlorophyll absorption, NIR plateau, and water absorption bands, injecting pre-symptomatic red-edge stress drops (~18.6%) inside localized fungal blight zones.
- **`src/data/pca_reducer.py`**:
  - `SpectralPCAReducer`: Flattens `(num_bands, H, W)` cubes into pixel spectral vectors, fits `sklearn.decomposition.PCA` (default `n_components=15`), and reshapes back to `(15, H, W)` while preserving $>95\%$ cumulative explained variance.
- **`src/data/dataset.py`**:
  - `HyperspectralPatchDataset`: PyTorch `Dataset` that extracts non-overlapping or strided 3D spatial-spectral patches `(1, C, patch_size, patch_size)` and matching 2D binary disease masks `(patch_size, patch_size)`.

---

## 2. Spectral-Spatial Deep Learning (`src/models/`)

- **`src/models/cnn3d.py`**:
  - `SpectralSpatial3DCNN`: Extracts local spectral-spatial features using two 3D convolutional blocks (`Conv3d` + `BatchNorm3d` + `GELU`), collapses the spectral axis via mean pooling to `(B, 32, H, W)`, and applies a `1x1` `Conv2d` segmentation head.
- **`src/models/vit.py`**:
  - `SpatialVisionTransformer`: Splits 2D feature maps into non-overlapping patches, projects them to `embed_dim=64`, adds learnable positional embeddings (with dynamic spatial interpolation), runs multi-head `nn.TransformerEncoder` self-attention, and upsamples back to `(B, embed_dim, H, W)`.
- **`src/models/hybrid.py`**:
  - `HybridSpectralSpatialViT`: Combines local 3D-CNN feature extraction with the global `SpatialVisionTransformer` branch via feature concatenation/fusion and a 2D segmentation classifier.
- **`src/models/train.py` & `src/models/metrics.py`**:
  - CPU-friendly training pipeline (`TrainConfig`, `train_one_epoch`, `evaluate_model`, `run_training`) using `AdamW` and `CrossEntropyLoss`, paired with binary segmentation metrics (`compute_confusion_counts`, `iou_score`, `dice_score`, `evaluate_segmentation`).
- **`src/models/checkpoint.py`**:
  - `save_model_checkpoint` & `load_model_checkpoint`: Persists and reconstructs `HybridSpectralSpatialViT` or `SpectralSpatial3DCNN` checkpoints (`.pt` files) along with training metadata.
- **`src/models/benchmark.py`**:
  - `count_trainable_parameters`, `benchmark_model`, and `compare_architectures`: Profiles parameter counts, CPU forward-pass latency (`min`/`mean`/`max` ms), and segmentation accuracy side by side.

---

## 3. Biophysical & Geospatial Analysis (`src/analysis/`)

- **`src/analysis/spectral_index.py`**:
  - `compute_ndvi`: Computes Normalized Difference Vegetation Index `(NIR - Red) / (NIR + Red + eps)` using bands closest to 670 nm and 800 nm.
  - `compute_red_edge_stress`: Computes `(NIR - RedEdge) / (NIR + RedEdge + eps)` around 710 nm to detect early chlorophyll degradation prior to visible wilting.
  - `summarize_field_health`: Computes infected pixel counts, infected percentage, infected area in hectares, mean disease probability, healthy vs. infected NDVI contrast, and overall severity level (`Low`, `Moderate`, `High`).
- **`src/analysis/geospatial.py`**:
  - `raster_to_geogrid`: Converts 2D probability, NDVI, and stress rasters into georeferenced records (`lat`, `lon`, `elevation`, `risk_label`, `color_rgba`) ready for PyDeck 3D `ColumnLayer` rendering.
  - `simulate_temporal_spread`: Models multi-day fungal blight expansion across forecast horizons (default `[0, 3, 7, 14]` days).

---

## 4. Sliding-Window Tiling & FastAPI (`src/api/`)

- **`src/api/tiling.py`**:
  - `TilingConfig` & `predict_full_raster`: Handles arbitrary raster dimensions by applying reflection/edge padding, slicing overlapping `(1, C, patch_size, patch_size)` tiles, running batched inference, and blending overlapping pixel probabilities cleanly back to `(H, W)`.
- **`src/api/schemas.py`**:
  - Pydantic v2 request/response schemas: `HealthResponse`, `ForecastRequest`, `ForecastResponse`, `ArchitectureBenchmarkItem`, and `BenchmarkResponse`.
- **`src/api/main.py`**:
  - FastAPI microservice exposing:
    - `GET /health` — Service and model readiness check.
    - `GET /api/v1/models` — Available architectures, default `pca_components`, and checkpoint status.
    - `POST /api/v1/forecast` — Runs the full pipeline and automatically loads trained demo weights via `get_model_for_inference(...)` when `checkpoints/<model_type>_demo.pt` is present.
    - `GET /api/v1/benchmark` — Runs side-by-side latency and accuracy benchmarking.

---

## 5. Streamlit + PyDeck 3D GIS Dashboard & CLI (`app.py`, `src/cli.py`)

- **`app.py`**:
  - Interactive Pure-Python 3D GIS dashboard featuring sidebar simulation controls, an optional FastAPI backend toggle with graceful local fallback, KPI metric cards, a PyDeck WebGL 3D `ColumnLayer` map with hover tooltips, CSV geogrid export, and an interactive **Model Benchmark** tab (`st.tabs`).
- **`src/cli.py`**:
  - Command-line interface runnable via `python -m src.cli` supporting `--height`, `--width`, `--bands`, `--pca-components`, `--model`, `--threshold`, `--seed`, and `--output` for JSON report generation.
