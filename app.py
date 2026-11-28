"""TerraSpectra: 3D Geospatial Crop Disease Forecasting Dashboard.

Built with 100% Pure Python using Streamlit and PyDeck for GPU-accelerated
3D geospatial rendering, biophysical spectral index analysis, and temporal
epidemic spread modeling.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch.nn as nn

from src.analysis.geospatial import raster_to_geogrid, simulate_temporal_spread
from src.analysis.spectral_index import (
    compute_ndvi,
    compute_red_edge_stress,
    compute_spectral_signatures,
    summarize_field_health,
)
from src.api.tiling import TilingConfig, predict_full_raster
from src.data.cube_loader import (
    HyperspectralCube,
    generate_synthetic_hyperspectral_cube,
    load_cube_from_bytes,
)
from src.data.pca_reducer import SpectralPCAReducer
from src.models.benchmark import compare_architectures
from src.models.cnn3d import SpectralSpatial3DCNN
from src.models.hybrid import HybridSpectralSpatialViT

# Global model cache to avoid re-instantiation overhead
_MODEL_CACHE: Dict[str, nn.Module] = {}


def get_cached_model(model_type: str) -> nn.Module:
    """Retrieve or instantiate requested neural model in evaluation mode."""
    if model_type not in _MODEL_CACHE:
        if model_type == "hybrid":
            model = HybridSpectralSpatialViT(
                in_channels=1,
                cnn_features=32,
                vit_embed_dim=64,
                patch_size=4,
                num_classes=2,
            )
        elif model_type == "cnn3d":
            model = SpectralSpatial3DCNN(
                in_channels=1,
                num_classes=2,
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")
        model.eval()
        _MODEL_CACHE[model_type] = model
    return _MODEL_CACHE[model_type]


def _run_pipeline_on_cube(
    cube: HyperspectralCube,
    model_type: str = "hybrid",
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run biophysical index extraction, PCA, inference, and geospatial analysis on a cube."""
    # Biophysical spectral indices
    ndvi_map = compute_ndvi(cube)
    stress_map = compute_red_edge_stress(cube)

    # Spectral PCA dimensionality reduction (cap components to available bands)
    n_components = min(15, cube.num_bands)
    reducer = SpectralPCAReducer(n_components=n_components, random_state=seed)
    reduced_raster = reducer.fit_transform(cube)

    # Neural inference via overlapping spatial tiling
    model = get_cached_model(model_type)
    tiling_cfg = TilingConfig(patch_size=16, stride=8, batch_size=8, device="cpu")
    prob_map = predict_full_raster(model, reduced_raster, config=tiling_cfg)

    # Temporal epidemic spread simulation across forecast horizons
    temporal_frames = simulate_temporal_spread(
        base_prob_map=prob_map,
        days=[0, 3, 7, 14],
        growth_rate=0.06,
    )

    # Field health summarization for base detection
    health_summary = summarize_field_health(
        prob_map=prob_map,
        ndvi_map=ndvi_map,
        threshold=threshold,
        pixel_area_sq_m=1.0,
    )

    # Convert to georeferenced 3D grid records
    geogrid_records = raster_to_geogrid(
        prob_map=prob_map,
        ndvi_map=ndvi_map,
        stress_map=stress_map,
        threshold=threshold,
    )

    return {
        "cube": cube,
        "prob_map": prob_map,
        "ndvi_map": ndvi_map,
        "stress_map": stress_map,
        "temporal_frames": temporal_frames,
        "health_summary": health_summary,
        "geogrid_records": geogrid_records,
    }


def run_dashboard_pipeline(
    model_type: str = "hybrid",
    size: int = 32,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end hyperspectral data generation, inference, and analysis.

    Pure-Python pipeline callable both inside the Streamlit app and directly in
    automated unit tests.

    Args:
        model_type: Model architecture ('hybrid' or 'cnn3d').
        size: Spatial raster extent (height and width in pixels).
        threshold: Blight classification probability threshold in [0.0, 1.0].
        seed: Random seed for synthetic cube generation.

    Returns:
        Dictionary containing:
            - 'prob_map': 2D float32 array of base disease probabilities
            - 'ndvi_map': 2D float32 array of NDVI values
            - 'stress_map': 2D float32 array of Red-Edge Stress values
            - 'temporal_frames': Dict mapping day (int) to 2D probability array
            - 'health_summary': Diagnostic dictionary from summarize_field_health
            - 'geogrid_records': List of georeferenced pixel records for PyDeck
    """
    cube, _ = generate_synthetic_hyperspectral_cube(
        height=size,
        width=size,
        num_bands=224,
        disease_zone_radius=max(4, size // 8),
        seed=seed,
    )
    return _run_pipeline_on_cube(
        cube=cube,
        model_type=model_type,
        threshold=threshold,
        seed=seed,
    )


def _call_fastapi_backend(
    api_url: str,
    model_type: str,
    size: int,
    threshold: float,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Attempt to call the FastAPI forecast endpoint and return parsed JSON.

    Uses only the Python standard library (``urllib.request`` + ``json``) so
    no extra dependencies are needed.

    Args:
        api_url: Full URL to the FastAPI ``/api/v1/forecast`` endpoint.
        model_type: ``"hybrid"`` or ``"cnn3d"``.
        size: Raster height/width in pixels.
        threshold: Blight probability classification threshold.
        seed: Random seed for reproducible synthetic generation.

    Returns:
        Parsed response dict on success, or ``None`` if the server is
        unavailable / returns a non-200 status.
    """
    payload = {
        "height": size,
        "width": size,
        "bands": 224,
        "pca_components": 15,
        "model_type": model_type,
        "disease_threshold": threshold,
        "pixel_area_sq_m": 1.0,
        "seed": seed,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError):
        pass
    return None


def main() -> None:
    """Render the interactive Streamlit + PyDeck geospatial dashboard."""
    st.set_page_config(
        page_title="TerraSpectra 3D GIS",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🛰️ TerraSpectra: 3D Geospatial Crop Disease Forecasting")
    st.caption(
        "100% Pure-Python Geospatial AI • NASA Hyperion / ESA Sentinel Simulation • "
        "3D Spectral-Spatial ViT + PyDeck WebGL"
    )

    # ==========================
    # Sidebar Controls
    # ==========================
    with st.sidebar:
        st.header("⚙️ Simulation & Model Controls")

        data_source = st.radio(
            "Data Source",
            options=["Synthetic Farm Cube", "Upload .npy / .npz Cube"],
            index=0,
            help="Choose between synthetic Hyperion simulation or your own 3D hyperspectral datacube.",
        )

        uploaded_file = None
        if data_source == "Upload .npy / .npz Cube":
            uploaded_file = st.file_uploader(
                "Upload Hyperspectral Cube (.npy / .npz)",
                type=["npy", "npz"],
                help="3D array (C, H, W) in .npy format or .npz with 'data' (and optional 'wavelengths_nm').",
            )

        model_type = st.selectbox(
            "Neural Architecture",
            options=["hybrid", "cnn3d"],
            index=0,
            help="Hybrid Spectral-Spatial ViT combines 3D-CNN with multi-head attention.",
        )

        size = st.select_slider(
            "Spatial Resolution (px)",
            options=[32, 64],
            value=32,
            help="Field dimension in pixels (e.g., 32x32 = 1,024 spatial survey points).",
        )

        threshold = st.slider(
            "Disease Probability Threshold",
            min_value=0.2,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help="Decision cutoff for classifying a canopy pixel as infected.",
        )

        selected_day = st.select_slider(
            "Forecast Timeline (Days)",
            options=[0, 3, 7, 14],
            value=0,
            help="Scrub through temporal disease spread progression.",
        )

        seed = st.number_input(
            "Random Simulation Seed",
            min_value=1,
            max_value=9999,
            value=42,
            step=1,
            help="Reproducibility seed for synthetic hyperspectral imaging spectrometer acquisition.",
        )

        st.markdown("---")
        st.subheader("🔌 Backend Configuration")

        use_fastapi = st.toggle(
            "Use FastAPI Backend",
            value=False,
            help=(
                "When enabled, the dashboard calls the TerraSpectra FastAPI microservice "
                "instead of running the pipeline locally.  Falls back to the local "
                "Python pipeline if the server is unreachable."
            ),
        )

        api_url = st.text_input(
            "FastAPI Forecast URL",
            value="http://127.0.0.1:8000/api/v1/forecast",
            disabled=not use_fastapi,
            help="URL of the POST /api/v1/forecast endpoint (requires the FastAPI server to be running).",
        )

    # ==========================
    # Execute Pipeline
    # ==========================
    if data_source == "Upload .npy / .npz Cube":
        if uploaded_file is None:
            st.info("👆 Upload a `.npy` or `.npz` hyperspectral cube in the sidebar to run 3D GIS forecasting.")
            return
        try:
            uploaded_cube = load_cube_from_bytes(
                raw_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
            )
        except ValueError as exc:
            st.error(f"❌ Invalid uploaded datacube: {exc}")
            return

        with st.spinner(f"Processing uploaded cube '{uploaded_file.name}' ({uploaded_cube.shape})..."):
            results = _run_pipeline_on_cube(
                cube=uploaded_cube,
                model_type=model_type,
                threshold=threshold,
                seed=int(seed),
            )
    else:
        with st.spinner("Processing Hyperspectral Datacube & Running 3D-ViT Inference..."):
            api_response: Optional[Dict[str, Any]] = None

            if use_fastapi:
                api_response = _call_fastapi_backend(
                    api_url=api_url,
                    model_type=model_type,
                    size=size,
                    threshold=threshold,
                    seed=int(seed),
                )
                if api_response is None:
                    st.warning(
                        "⚠️ **FastAPI backend unreachable** — falling back to the local "
                        "Python pipeline.  Start the server with:\n\n"
                        "```\n.venv\\Scripts\\uvicorn.exe src.api.main:app --reload --port 8000\n```"
                    )

            # Use local pipeline if FastAPI is disabled or server was unreachable
            if api_response is None:
                results = run_dashboard_pipeline(
                    model_type=model_type,
                    size=size,
                    threshold=threshold,
                    seed=int(seed),
                )
            else:
                # Build a results dict compatible with the rest of the dashboard from
                # the FastAPI JSON response.  The API returns flat prob / ndvi maps;
                # we reconstruct numpy arrays and derive temporal frames locally.
                import numpy as _np  # local to avoid shadowing module-level np

                prob_flat = api_response.get("probability_map", [])
                ndvi_flat = api_response.get("ndvi_map", [])
                stress_flat = api_response.get("red_edge_stress_map", [])

                prob_map_api = _np.array(prob_flat, dtype=_np.float32)
                ndvi_map_api = _np.array(ndvi_flat, dtype=_np.float32)
                stress_map_api = _np.array(stress_flat, dtype=_np.float32)

                temporal_frames = simulate_temporal_spread(
                    base_prob_map=prob_map_api,
                    days=[0, 3, 7, 14],
                    growth_rate=0.06,
                )
                health_summary_api = summarize_field_health(
                    prob_map=prob_map_api,
                    ndvi_map=ndvi_map_api,
                    threshold=threshold,
                    pixel_area_sq_m=1.0,
                )
                geogrid_records_api = raster_to_geogrid(
                    prob_map=prob_map_api,
                    ndvi_map=ndvi_map_api,
                    stress_map=stress_map_api,
                    threshold=threshold,
                )
                results = {
                    "prob_map": prob_map_api,
                    "ndvi_map": ndvi_map_api,
                    "stress_map": stress_map_api,
                    "temporal_frames": temporal_frames,
                    "health_summary": health_summary_api,
                    "geogrid_records": geogrid_records_api,
                }
                st.success("✅ Results fetched from FastAPI backend.")

    # ==========================
    # Main View Tabs
    # ==========================
    tab_map, tab_benchmark = st.tabs(["🗺️ 3D Disease Map", "⚡ Model Benchmark"])

    with tab_map:
        # Extract target day's spatial frame
        current_prob_map = results["temporal_frames"][selected_day]
        ndvi_map = results["ndvi_map"]
        stress_map = results["stress_map"]

        # Recompute metrics and geogrid for selected timeline day
        health_summary = summarize_field_health(
            prob_map=current_prob_map,
            ndvi_map=ndvi_map,
            threshold=threshold,
            pixel_area_sq_m=1.0,
        )
        geogrid_records = raster_to_geogrid(
            prob_map=current_prob_map,
            ndvi_map=ndvi_map,
            stress_map=stress_map,
            threshold=threshold,
        )
        df_grid = pd.DataFrame(geogrid_records)

        # ==========================
        # 4 KPI Metric Cards
        # ==========================
        col1, col2, col3, col4 = st.columns(4)

        severity_color = {
            "Low": "🟢",
            "Moderate": "🟡",
            "High": "🔴",
        }.get(health_summary["severity_level"], "⚪")

        col1.metric(
            label="Severity Level",
            value=f"{severity_color} {health_summary['severity_level']}",
            delta=f"Day {selected_day} Forecast",
        )
        col2.metric(
            label="Infected Area (%)",
            value=f"{health_summary['infected_percentage']:.1f}%",
            delta=f"{health_summary['infected_pixels']} / {health_summary['total_pixels']} px",
            delta_color="inverse",
        )
        col3.metric(
            label="Infected Hectares",
            value=f"{health_summary['infected_area_hectares']:.3f} ha",
            help="Physical ground area affected (1 ha = 10,000 m²).",
        )
        col4.metric(
            label="Mean Disease Risk",
            value=f"{health_summary['mean_disease_probability']:.2%}",
            help="Canopy-wide average fungal blight risk score.",
        )

        st.markdown("---")

        # ==========================
        # Interactive PyDeck 3D Map
        # ==========================
        st.subheader(f"🗺️ 3D Geospatial Field Risk Map (Day {selected_day})")
        st.caption(
            "Interactive 3D WebGL column rendering. Elevation indicates disease probability "
            "(0-80m scaled height). Colors: Green = Healthy, Amber = Watch, Glowing Red = Blight Outbreak."
        )

        column_layer = pdk.Layer(
            "ColumnLayer",
            data=df_grid,
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1.0,
            radius=8,
            get_fill_color="color_rgba",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

        origin_lat = df_grid["lat"].mean()
        origin_lon = df_grid["lon"].mean()

        view_state = pdk.ViewState(
            latitude=origin_lat,
            longitude=origin_lon,
            zoom=16.5,
            pitch=48,
            bearing=18,
        )

        tooltip = {
            "html": """
            <div style="background-color: #1a1c23; padding: 10px; border-radius: 6px; font-family: monospace; border: 1px solid #333;">
                <b style="color: #61afef;">TerraSpectra Survey Point</b><br/>
                <b>Status:</b> {risk_label}<br/>
                <b>Blight Risk:</b> {probability}<br/>
                <b>NDVI:</b> {ndvi}<br/>
                <b>Red-Edge Stress:</b> {stress}<br/>
                <b>Coord:</b> ({lat}, {lon})
            </div>
            """,
            "style": {"color": "white"},
        }

        deck = pdk.Deck(
            layers=[column_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        )

        st.pydeck_chart(deck, use_container_width=True)

        # ==========================
        # Spectral Reflectance Signature (400-2500 nm)
        # ==========================
        if "cube" in results:
            st.markdown("---")
            st.subheader("📈 Spectral Reflectance Signature (400–2500 nm)")
            st.caption(
                "Mean canopy reflectance profile comparing Healthy vs. Infected pixels across all spectral bands."
            )
            signatures = compute_spectral_signatures(
                cube=results["cube"],
                prob_map=current_prob_map,
                threshold=threshold,
            )
            df_spectra = pd.DataFrame(
                {
                    "Healthy": signatures["healthy_mean"],
                    "Infected": signatures["infected_mean"],
                },
                index=pd.Index(signatures["wavelengths_nm"], name="Wavelength (nm)"),
            )
            st.line_chart(df_spectra)

        # ==========================
        # Biophysical Indices & Export
        # ==========================
        st.markdown("---")
        info_col1, info_col2 = st.columns([2, 1])

        with info_col1:
            st.subheader("🌿 Biophysical Canopy Health Analysis")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            comp_col1.metric("Healthy Canopy NDVI", f"{health_summary['mean_ndvi_healthy']:.3f}")
            comp_col2.metric("Infected Canopy NDVI", f"{health_summary['mean_ndvi_infected']:.3f}")
            ndvi_delta = health_summary["mean_ndvi_healthy"] - health_summary["mean_ndvi_infected"]
            comp_col3.metric("Foliar Contrast Delta", f"{ndvi_delta:.3f}")

            st.info(
                "💡 **Pre-symptomatic Detection**: Note that visible NDVI remains relatively resilient "
                "in early phases, while the 3D-ViT leverages continuous red-edge transition bands (700-750nm) "
                "to detect cellular collapse before physical wilting or chlorosis emerges."
            )

        with info_col2:
            st.subheader("📥 Export Georeferenced Risk Data")
            st.write("Export detailed 3D column records for GIS integration (QGIS, ArcGIS, GeoJSON).")
            csv_bytes = df_grid.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Survey Grid (CSV)",
                data=csv_bytes,
                file_name=f"terraspectra_grid_day_{selected_day}_{model_type}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_benchmark:
        st.subheader("⚡ Model Performance & Latency Benchmark")
        st.caption(
            "Side-by-side performance profiling: SpectralSpatial3DCNN vs. "
            "HybridSpectralSpatialViT on inference latency, parameter counts, and segmentation accuracy."
        )

        if st.button("Run 3D-CNN vs Hybrid ViT Benchmark", key="run_benchmark_btn"):
            with st.spinner("Benchmarking architectures (warmup + timed runs)..."):
                bench_results = compare_architectures(
                    height=32,
                    width=32,
                    pca_components=15,
                    seed=int(seed),
                )

                bench_rows = []
                for r in bench_results:
                    m = r.get("metrics") or {}
                    bench_rows.append({
                        "Architecture": r["model_name"],
                        "Trainable Params": f"{r['trainable_params']:,}",
                        "Mean Latency (ms)": f"{r['mean_latency_ms']:.2f}",
                        "Min Latency (ms)": f"{r['min_latency_ms']:.2f}",
                        "Max Latency (ms)": f"{r['max_latency_ms']:.2f}",
                        "IoU": f"{m.get('iou', 0.0):.4f}",
                        "Dice": f"{m.get('dice', 0.0):.4f}",
                        "Precision": f"{m.get('precision', 0.0):.4f}",
                        "Recall": f"{m.get('recall', 0.0):.4f}",
                        "Pixel Accuracy": f"{m.get('pixel_accuracy', 0.0):.4f}",
                    })

                st.markdown("### 📋 Architecture Comparison Table")
                st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

                st.markdown("### 📊 Mean Inference Latency (ms)")
                df_latency = pd.DataFrame(
                    {"Mean Latency (ms)": [r["mean_latency_ms"] for r in bench_results]},
                    index=[r["model_name"] for r in bench_results],
                )
                st.bar_chart(df_latency)


if __name__ == "__main__":
    main()
