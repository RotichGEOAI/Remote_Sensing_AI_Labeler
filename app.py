from __future__ import annotations

import io
import json
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from skimage import measure, morphology

try:
    import rasterio
except ImportError:  # pragma: no cover - optional at runtime
    rasterio = None

from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Remote Sensing AI Labeller",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_CLASSES = [
    {"name": "Building", "color": "#F97316", "hotkey": "B"},
    {"name": "Vegetation", "color": "#22C55E", "hotkey": "V"},
    {"name": "Water", "color": "#38BDF8", "hotkey": "W"},
    {"name": "Road", "color": "#EAB308", "hotkey": "R"},
    {"name": "Bare soil", "color": "#A78BFA", "hotkey": "S"},
]


def init_state() -> None:
    defaults = {
        "classes": DEFAULT_CLASSES.copy(),
        "annotations": [],
        "suggestions": [],
        "audit": [],
        "scene": None,
        "scene_name": "Synthetic coastal scene",
        "render_mode": "True colour",
        "active_class": "Building",
        "canvas_key": 0,
        "canvas_object_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def synthetic_scene() -> dict[str, Any]:
    """Create a repeatable small multispectral scene for demos and tests."""
    rng = np.random.default_rng(42)
    height, width = 420, 640
    y, x = np.mgrid[0:height, 0:width]
    water = ((x - 115) / 145) ** 2 + ((y - 245) / 180) ** 2 < 1
    vegetation = np.sin(x / 24) + np.cos(y / 37) > 0.15
    vegetation &= ~water
    buildings = np.zeros((height, width), dtype=bool)
    for left, top, w, h in [(300, 62, 82, 58), (430, 115, 108, 72), (270, 280, 145, 62), (500, 300, 74, 82)]:
        buildings[top : top + h, left : left + w] = True
    roads = (np.abs(y - (0.52 * x + 35)) < 12) | (np.abs(y - (-0.40 * x + 410)) < 10)
    red = np.where(water, 0.08, np.where(vegetation, 0.16, np.where(buildings, 0.58, 0.35)))
    green = np.where(water, 0.20, np.where(vegetation, 0.62, np.where(buildings, 0.48, 0.35)))
    blue = np.where(water, 0.65, np.where(vegetation, 0.22, np.where(buildings, 0.38, 0.30)))
    nir = np.where(water, 0.10, np.where(vegetation, 0.82, np.where(buildings, 0.52, 0.38)))
    red[roads], green[roads], blue[roads], nir[roads] = 0.42, 0.42, 0.42, 0.44
    stack = np.clip(np.stack([red, green, blue, nir]) + rng.normal(0, 0.035, (4, height, width)), 0, 1)
    return {"array": stack.astype("float32"), "crs": None, "transform": None, "name": "Synthetic coastal scene"}


def load_uploaded(file: Any) -> dict[str, Any]:
    name = file.name.lower()
    raw = file.getvalue()
    if name.endswith((".tif", ".tiff")) and rasterio:
        with rasterio.MemoryFile(raw).open() as dataset:
            array = dataset.read().astype("float32")
            if array.shape[0] == 1:
                array = np.repeat(array, 4, axis=0)
            return {
                "array": normalise_bands(array),
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": dataset.transform,
                "name": file.name,
            }
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.asarray(image).astype("float32") / 255
    if rgb.shape[2] == 3:
        rgb = np.concatenate([rgb, rgb[:, :, 1:2]], axis=2)
    return {"array": np.moveaxis(rgb, 2, 0), "crs": None, "transform": None, "name": file.name}


def normalise_bands(array: np.ndarray) -> np.ndarray:
    output = np.zeros_like(array, dtype="float32")
    for index, band in enumerate(array):
        low, high = np.nanpercentile(band, [2, 98])
        output[index] = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
    if output.shape[0] < 4:
        output = np.concatenate([output, np.repeat(output[-1:], 4 - output.shape[0], axis=0)])
    return output


def rgb_image(scene: dict[str, Any], mode: str) -> np.ndarray:
    bands = scene["array"]
    if mode == "False colour (NIR-R-G)":
        channels = [3, 0, 1]
    elif mode == "NDVI":
        ndvi = (bands[3] - bands[0]) / (bands[3] + bands[0] + 1e-6)
        return ((ndvi + 1) / 2 * 255).clip(0, 255).astype("uint8")
    else:
        channels = [0, 1, 2]
    return (np.moveaxis(bands[channels], 0, 2) * 255).clip(0, 255).astype("uint8")


def ndvi(scene: dict[str, Any]) -> np.ndarray:
    bands = scene["array"]
    return (bands[3] - bands[0]) / (bands[3] + bands[0] + 1e-6)


def record(event: str, detail: str) -> None:
    st.session_state.audit.insert(
        0, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "event": event, "detail": detail}
    )


def run_ai_assist(scene: dict[str, Any], target: str, threshold: float, min_area: int) -> list[dict[str, Any]]:
    bands = scene["array"]
    if target == "Vegetation":
        score = ndvi(scene)
        mask = score > threshold
    elif target == "Water":
        score = bands[2] - bands[3]
        mask = score > threshold - 0.3
    elif target == "Building":
        score = bands[0] * 0.55 + bands[1] * 0.45
        mask = score > threshold
    elif target == "Road":
        score = np.std(bands[:3], axis=0)
        mask = score < threshold
    else:
        score = bands[0] - bands[3]
        mask = score > threshold - 0.3
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    suggestions = []
    for region in measure.regionprops(measure.label(mask)):
        if region.area < min_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        suggestions.append(
            {
                "id": str(uuid.uuid4())[:8],
                "class": target,
                "geometry": "rectangle",
                "x": int(min_col),
                "y": int(min_row),
                "width": int(max_col - min_col),
                "height": int(max_row - min_row),
                "area_px": int(region.area),
                "confidence": float(min(0.99, 0.55 + region.area / (scene["array"].shape[1] * scene["array"].shape[2]) * 12)),
                "status": "Suggested",
            }
        )
    return sorted(suggestions, key=lambda item: item["confidence"], reverse=True)[:80]


def export_geojson(scene: dict[str, Any], annotations: list[dict[str, Any]]) -> bytes:
    features = []
    height, width = scene["array"].shape[1:]
    for item in annotations:
        x, y, w, h = item["x"], item["y"], item["width"], item["height"]
        features.append(
            {
                "type": "Feature",
                "properties": {k: v for k, v in item.items() if k not in {"x", "y", "width", "height"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [x / width, y / height],
                        [(x + w) / width, y / height],
                        [(x + w) / width, (y + h) / height],
                        [x / width, (y + h) / height],
                        [x / width, y / height],
                    ]],
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode()


def png_bytes(image_array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_array).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


init_state()

with st.sidebar:
    st.markdown("## 🛰️ Scene setup")
    uploaded = st.file_uploader("Upload imagery", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if st.button("Use synthetic demo", use_container_width=True):
        st.session_state.scene = synthetic_scene()
        st.session_state.scene_name = "Synthetic coastal scene"
        st.session_state.annotations = []
        st.session_state.suggestions = []
        record("Scene loaded", "Synthetic coastal scene")
    if uploaded is not None and (st.session_state.scene is None or st.session_state.scene_name != uploaded.name):
        st.session_state.scene = load_uploaded(uploaded)
        st.session_state.scene_name = uploaded.name
        st.session_state.annotations = []
        st.session_state.suggestions = []
        record("Scene loaded", uploaded.name)
    st.divider()
    st.markdown("### Classes")
    for item in st.session_state.classes:
        st.markdown(
            f'<span style="display:inline-block;width:11px;height:11px;background:{item["color"]};'
            f'border-radius:50%;margin-right:7px"></span>{item["name"]} <small>({item["hotkey"]})</small>',
            unsafe_allow_html=True,
        )
    with st.expander("Add class"):
        class_name = st.text_input("Class name", key="new_class_name")
        class_color = st.color_picker("Colour", "#EC4899", key="new_class_color")
        if st.button("Add class") and class_name.strip():
            st.session_state.classes.append({"name": class_name.strip(), "color": class_color, "hotkey": class_name[0].upper()})
            record("Class added", class_name.strip())
            st.rerun()

if st.session_state.scene is None:
    st.session_state.scene = synthetic_scene()
    record("Scene loaded", "Synthetic coastal scene")

scene = st.session_state.scene
height, width = scene["array"].shape[1:]
st.title("Remote Sensing AI Labeller")
st.caption("A transparent, human-in-the-loop workspace for turning imagery into training-ready labels.")

metrics = st.columns(4)
metrics[0].metric("Scene", st.session_state.scene_name[:22])
metrics[1].metric("Resolution", f"{width} × {height}")
metrics[2].metric("Bands", scene["array"].shape[0])
metrics[3].metric("Labels", len(st.session_state.annotations))

tab_label, tab_ai, tab_review, tab_export = st.tabs(["🖍️ Annotate", "✨ AI assist", "✅ Review", "📦 Export"])

with tab_label:
    control_col, image_col = st.columns([1, 3])
    with control_col:
        st.markdown("#### Rendering")
        st.session_state.render_mode = st.selectbox(
            "Visualisation", ["True colour", "False colour (NIR-R-G)", "NDVI"], label_visibility="collapsed"
        )
        st.markdown("#### Annotation tool")
        tool = st.selectbox("Geometry", ["rect", "polygon", "circle", "point"], format_func=lambda x: x.title())
        class_names = [item["name"] for item in st.session_state.classes]
        st.session_state.active_class = st.selectbox("Active class", class_names, index=class_names.index(st.session_state.active_class) if st.session_state.active_class in class_names else 0)
        st.info("Draw directly on the image. Each completed shape is stored as a label.")
        if st.button("Clear all labels", type="secondary", use_container_width=True):
            st.session_state.annotations = []
            st.session_state.canvas_key += 1
            st.session_state.canvas_object_count = 0
            record("Labels cleared", "All manual labels removed")
            st.rerun()
    with image_col:
        preview = rgb_image(scene, st.session_state.render_mode)
        if preview.ndim == 2:
            preview = np.stack([preview] * 3, axis=2)
        display = Image.fromarray(preview).resize((min(width, 900), min(height, 620)))
        canvas_result = st_canvas(
            fill_color="rgba(34,197,94,0.22)",
            stroke_width=3,
            stroke_color="#22C55E",
            background_image=display,
            update_streamlit=True,
            height=display.height,
            width=display.width,
            drawing_mode=tool,
            key=f"canvas_{st.session_state.canvas_key}_{st.session_state.render_mode}",
        )
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            objects = canvas_result.json_data["objects"]
            known = st.session_state.canvas_object_count
            for obj in objects[known:]:
                scale_x, scale_y = width / display.width, height / display.height
                item = {
                    "id": str(uuid.uuid4())[:8],
                    "class": st.session_state.active_class,
                    "geometry": obj.get("type", tool),
                    "x": round(obj.get("left", 0) * scale_x),
                    "y": round(obj.get("top", 0) * scale_y),
                    "width": round(obj.get("width", 8) * scale_x),
                    "height": round(obj.get("height", 8) * scale_y),
                    "area_px": round(max(1, obj.get("width", 8) * obj.get("height", 8) * scale_x * scale_y)),
                    "confidence": 1.0,
                    "status": "Manual",
                }
                st.session_state.annotations.append(item)
                record("Manual label", f'{item["class"]} ({item["geometry"]})')
            st.session_state.canvas_object_count = len(objects)

with tab_ai:
    st.markdown("### Generate candidate labels")
    st.write("AI assist uses spectral indices and connected components to propose regions. Review every suggestion before export.")
    ai_col1, ai_col2, ai_col3 = st.columns(3)
    with ai_col1:
        target = st.selectbox("Target class", [item["name"] for item in st.session_state.classes], key="ai_target")
    with ai_col2:
        threshold = st.slider("Sensitivity", 0.1, 0.95, 0.55, 0.01)
    with ai_col3:
        min_area = st.slider("Minimum region (px)", 20, 3000, 180, 20)
    if st.button("Run AI assist", type="primary"):
        st.session_state.suggestions = run_ai_assist(scene, target, threshold, min_area)
        record("AI assist", f"{len(st.session_state.suggestions)} {target} candidates generated")
    if st.session_state.suggestions:
        st.success(f"{len(st.session_state.suggestions)} candidates ready for review.")
        suggestion_df = pd.DataFrame(st.session_state.suggestions)
        st.dataframe(suggestion_df[["id", "class", "area_px", "confidence", "status"]], use_container_width=True, hide_index=True)
        accept_col, reject_col = st.columns(2)
        with accept_col:
            if st.button("Accept all candidates", use_container_width=True):
                for item in st.session_state.suggestions:
                    item["status"] = "Accepted"
                st.session_state.annotations.extend(st.session_state.suggestions)
                record("AI labels accepted", f"{len(st.session_state.suggestions)} candidates")
                st.session_state.suggestions = []
                st.rerun()
        with reject_col:
            if st.button("Reject all candidates", use_container_width=True):
                record("AI labels rejected", f"{len(st.session_state.suggestions)} candidates")
                st.session_state.suggestions = []
                st.rerun()

with tab_review:
    st.markdown("### Label quality control")
    if st.session_state.annotations:
        labels = pd.DataFrame(st.session_state.annotations)
        chart_col, table_col = st.columns([1, 2])
        with chart_col:
            counts = labels["class"].value_counts().rename_axis("class").reset_index(name="labels")
            st.plotly_chart(px.bar(counts, x="class", y="labels", color="class", height=280), use_container_width=True)
        with table_col:
            st.data_editor(
                labels[["id", "class", "geometry", "area_px", "confidence", "status"]],
                use_container_width=True,
                hide_index=True,
                disabled=["id", "geometry", "area_px", "confidence", "status"],
            )
        st.caption(f"Total labelled area: {labels['area_px'].sum():,} px · Mean confidence: {labels['confidence'].mean():.0%}")
    else:
        st.info("No labels yet. Draw on the Annotate tab or accept AI suggestions.")
    with st.expander("Audit trail"):
        st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True, hide_index=True)

with tab_export:
    st.markdown("### Export training labels")
    st.write("Exports use image-relative coordinates for portability. GeoTIFF scenes also retain their source CRS in the session metadata.")
    if not st.session_state.annotations:
        st.warning("Add at least one label before exporting.")
    else:
        annotations_df = pd.DataFrame(st.session_state.annotations)
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download labels as CSV",
                annotations_df.to_csv(index=False).encode(),
                file_name="labels.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_col2:
            st.download_button(
                "Download labels as GeoJSON",
                export_geojson(scene, st.session_state.annotations),
                file_name="labels.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
        st.download_button(
            "Download scene preview",
            png_bytes(rgb_image(scene, "True colour")),
            file_name="scene-preview.png",
            mime="image/png",
        )
