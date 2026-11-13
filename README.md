# TerraSpectra: Hyperspectral Crop Disease Forecasting

TerraSpectra is an advanced geospatial AI framework for early-stage crop disease detection and pre-symptomatic fungal blight forecasting using continuous imaging spectrometer datacubes and hybrid deep learning models.

---

## 100% Pure-Python Architecture

TerraSpectra employs a **100% Pure-Python stack**, eliminating Node.js, npm, and separate JavaScript frontend runtimes (such as React and Deck.gl wrappers) in favor of a unified, high-performance Python ecosystem.

```
+-----------------------------------------------------------------------------------+
|                           100% PURE-PYTHON ARCHITECTURE                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-------------------------------------+   +-----------------------------------+  |
|  |     Interactive Geospatial UI       |   |       High-Throughput API         |  |
|  |        Streamlit + PyDeck           |   |       FastAPI + Uvicorn           |  |
|  |  (GPU-accelerated WebGL layers,     |<->|   (REST endpoints, Pydantic v2,   |  |
|  |   spatial blight risk heatmap)      |   |    async inference streaming)     |  |
|  +-------------------------------------+   +-----------------------------------+  |
|                     ^                                        ^                    |
|                     |                                        |                    |
|                     +-------------------+--------------------+                    |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  |                             Deep Learning Engine                            |  |
|  |                         PyTorch 2.x + TorchVision                           |  |
|  |  - SpectralSpatial3DCNN: 3D spatial-spectral local feature extraction       |  |
|  |  - SpatialVisionTransformer: Multi-head attention for field-wide context   |  |
|  |  - HybridSpectralSpatialViT: End-to-end 3D-CNN + ViT fusion segmentation   |  |
|  +-----------------------------------------------------------------------------+  |
|                                         ^                                         |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  |                      Tier 1 Geospatial & Data Pipeline                      |  |
|  |                          NumPy + Scikit-Learn                               |  |
|  |  - HyperspectralCube: 224-band continuous sensor simulation (400-2500 nm)   |  |
|  |  - SpectralPCAReducer: Fast dimensionality reduction (>=95% variance)      |  |
|  |  - HyperspectralPatchDataset: 3D spatial-spectral patch slicing & batching  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Key Rationale & Benefits of Pure Python:
- **Zero-Serialization Overhead**: PyTorch inference output tensors and NumPy geospatial grids directly feed into PyDeck layer specifications without JSON round-tripping or cross-language marshaling.
- **Unified Dependency Management**: A single Python virtual environment (`requirements.txt` / `pyproject.toml`) manages the full application lifecycle. No `package.json`, Node.js engines, or npm builds required.
- **Scientific Geospatial Native**: Full compatibility with the Python scientific stack (NumPy, SciPy, PyTorch, Scikit-learn, GDAL/Shapely/PyProj).
- **Rapid Prototyping & Production Parity**: Interactive Streamlit dashboards share the exact model instances, weights, and preprocessors as the FastAPI production microservice.

---

## Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Frontend & Visualization** | **Streamlit** + **PyDeck** | Interactive field explorer, 3D WebGL crop health layers, spectral profile inspection |
| **Backend API** | **FastAPI** + **Uvicorn** + **Pydantic** | High-performance asynchronous inference endpoints and data validation |
| **Deep Learning** | **PyTorch** | 3D-CNN, Spatial Vision Transformer, and Hybrid neural architectures |
| **Geospatial & Spectral Preprocessing** | **NumPy** + **Scikit-Learn** | 224-band datacube simulation, spectral curve modeling, and PCA dimensionality reduction |
| **Testing & CI** | **pytest** + **unittest** | Comprehensive unit, integration, and backpropagation test suites |

---

## Repository Structure

```
TerraSpectra/
├── .gitignore                      # Git ignore for venv, cache, and raw data
├── pyproject.toml                  # PEP 518/621 build configuration
├── requirements.txt                # 100% Pure-Python dependencies
├── README.md                       # Project architecture and documentation
├── src/
│   ├── __init__.py
│   ├── api/                        # FastAPI microservices & endpoint routers
│   │   └── __init__.py
│   ├── data/                       # Geospatial & hyperspectral data pipelines
│   │   ├── __init__.py
│   │   ├── cube_loader.py          # HyperspectralCube dataclass & synthetic sensor simulator
│   │   ├── pca_reducer.py          # SpectralPCAReducer (224 bands -> 15 components)
│   │   └── dataset.py              # HyperspectralPatchDataset for PyTorch DataLoader
│   ├── gis/                        # Geospatial coordinate systems & GeoJSON handlers
│   │   └── __init__.py
│   └── models/                     # Deep learning architectures
│       ├── __init__.py
│       ├── cnn3d.py                # SpectralSpatial3DCNN (3D Conv + spectral depth pooling)
│       ├── vit.py                  # SpatialVisionTransformer (Multi-head attention on 2D maps)
│       └── hybrid.py               # HybridSpectralSpatialViT (3D-CNN + ViT fusion)
└── tests/
    ├── __init__.py
    ├── test_api/                   # API test suite
    │   └── __init__.py
    ├── test_data/                  # Data ingestion & PCA compression tests
    │   ├── __init__.py
    │   └── test_cube_loader.py
    └── test_models/                # PyTorch model forward, backward & batching tests
        ├── __init__.py
        ├── test_cnn3d.py
        └── test_vit.py
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/TerraSpectra.git
   cd TerraSpectra
   ```

2. **Set up a Python 3.11+ virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows
   # source .venv/bin/activate # Linux / macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Test Suite

Execute the complete test suite across all data loaders, dimensionality reducers, and deep learning architectures:

```bash
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

Or using pytest:

```bash
.\.venv\Scripts\python.exe -m pytest -v
```
