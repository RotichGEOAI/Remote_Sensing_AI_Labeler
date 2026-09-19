# Remote Sensing AI Labeller

An end-to-end Streamlit workspace for rapidly creating training labels from satellite and aerial imagery.

## What it includes

- GeoTIFF, PNG, and JPG ingestion with multiband support
- True-colour, false-colour, and NDVI visualisation
- AI-assisted candidate segmentation using configurable spectral thresholds
- Interactive polygon, rectangle, and point annotation
- Class management with colour-coded labels
- Annotation review table and class distribution chart
- GeoJSON and CSV export, with a downloadable preview image
- Session audit trail for reproducibility
- Built-in synthetic scene so the workflow can be tried without data

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For GeoTIFF support, install the platform prerequisites required by `rasterio` if your Python distribution does not provide a wheel.

## Workflow

1. Load a scene from the sidebar or use the built-in demo.
2. Pick a rendering preset and inspect the scene statistics.
3. Choose a target class and run AI assist to generate candidate regions.
4. Draw labels over the image and accept or reject suggested candidates.
5. Review the annotation table and export labels for model training.

The AI assist module is intentionally transparent and deterministic: it uses image-derived indices and connected components rather than pretending to be a foundation model. It can be replaced by a remote inference endpoint without changing the labelling UI.
