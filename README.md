# 🛢️ Drilling Fluid Dashboard

Streamlit app to extract and visualize drilling fluid data from daily reports.

## Features
- PDF data extraction
- ML-based degradation prediction
- KPI charts (ROP, SCE, DSRE%, etc.)
- Filter by Well Name and Date Range
- Downloadable CSV output

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker build -t fluid-dashboard .
docker run -p 8501:8501 fluid-dashboard
```

## Deployment
Upload to Streamlit Cloud with:
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
