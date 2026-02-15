# 🌍 AQI Predictor - Air Quality Forecasting System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/ML-Scikit--learn%20|%20TensorFlow-orange.svg" alt="ML">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

**Real-time Air Quality Index (AQI) prediction system for Hyderabad, Sindh, Pakistan using machine learning and automated pipelines.**

Predict AQI levels for the next 72 hours with 95%+ accuracy using ensemble ML models, automated hourly data collection, and daily model retraining. Features interactive dashboard, health alerts, and SHAP explainability.

**Live App**: https://hyderabad-pearls-aqi-predictor.streamlit.app/

**Backend API (Render)**: https://aqi-predictor-4r2g.onrender.com

---

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Model Performance](#-model-performance)

---

## ✨ Features

### 🎯 Core Capabilities
- **72-Hour AQI Forecasting** - Hourly predictions with confidence scores
- **5 ML Models** - Random Forest, XGBoost, LightGBM, ElasticNet, TensorFlow NN
- **Automated Pipelines** - Hourly data collection, daily model training
- **Real-time Predictions** - Sub-second inference via FastAPI backend
- **Interactive Dashboard** - Streamlit UI with 6 pages of insights

### 🔬 Advanced Analytics
- **SHAP Explainability** - Understand which features drive predictions
- **Health Alerts** - 4-level alert system (Normal → Hazard)
- **Historical Analysis** - Trend visualization and pattern detection
- **Feature Engineering** - 50+ engineered features (lags, interactions, cyclical)

### 🚀 Production-Ready
- **100% Serverless** - GitHub Actions + Hopsworks
- **Model Registry** - Versioned models in Hopsworks
- **Feature Store** - 2+ years of observed historical data (no forecast data in training)
- **Caching** - Optimized response times (< 100ms)
- **Type Safety** - Automatic float32/float64 schema conversions

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (CI/CD)                   │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │ Feature Pipeline │ Hourly  │  Training Pipeline      │  │
│  │  (Collect Data)  │ ──────► │  (Train 5 Models Daily) │  │
│  └──────────────────┘         └─────────────────────────┘  │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
             ▼                              ▼
      ┌──────────────┐              ┌──────────────┐
      │  Hopsworks   │◄─────────────┤ Model Registry│
      │Feature Store │              └──────────────┘
      └──────┬───────┘
             │
             ▼
      ┌──────────────────────────────────────┐
      │         FastAPI Backend              │
      │  - Load Models from Registry         │
      │  - Generate Predictions (72h)        │
      │  - SHAP Explainability               │
      │  - Alert System                      │
      └──────────────┬───────────────────────┘
                     │
                     ▼
      ┌──────────────────────────────────────┐
      │       Streamlit Dashboard            │
      │  🏠 Home  📊 Forecast  ❤️ Health     │
      │  📈 History  ⚠️ Alerts  🤖 Models   │
      └──────────────────────────────────────┘
```

### Pipeline Flow

**0. Historical Backfill** (One-time setup)
   - **Script**: `src/backfill/backfill_historical_data.py`
   - Fetches 2 years of historical data (weather + observed AQI)
   - Deduplicates by `time`, sorts chronologically
   - Generates engineered features with lag computation (~17,520 records)
   - Stores in **`aqi_historical_features`** feature group (primary key: `time`)
   - Run once before first training

**1. Feature Pipeline** (Hourly at :42 past the hour)
   - **Splits into two independent tasks:**
   
   **A. Historical Features (Training Data)**
   - Fetches last 26 hours of observed data (weather + AQI)
   - Processes all 26 hours to compute 24h lag features correctly
   - Deduplicates by `time`, enforces chronological order
   - Inserts **only last 2 hours** of new timestamps (~2-10 rows)
   - Updates **`aqi_historical_features`** feature group
   - No forecast contamination - only real observed data
   
   **B. Forecast Weather Features (Inference Only)**
   - Fetches 3-day forecast weather (temperature, wind, etc.)
   - Drops pollutants, computes time-based features only
   - Stores in **`weather_forecast_features`** feature group
   - Updated hourly for next 72h inference

2. **Training Pipeline** (Daily @ 8:47 AM UTC)
   - Load 2+ years from **`aqi_historical_features`** only
   - Sort by `time` to enforce temporal order
   - Train 5 ML models with time-series cross-validation
   - Select best model (lowest validation RMSE)
   - **Print & export metrics table → `docs/model_metrics_table.csv`**
   - Register all models to Hopsworks Model Registry
   - Generate 72h forecast cache
   - Save artifacts to `models/cache/`

3. **Inference** (On-Demand)
   - Load best model from registry
   - Fetch last historical row → extract lag features
   - Fetch forecast weather from `weather_forecast_features` (or fallback to API)
   - Combine historical lags + forecast weather → predict
   - Serve via FastAPI
   - Display in Streamlit

---

## 🛠 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **ML Frameworks** | Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit, Plotly |
| **Feature Store** | Hopsworks (2 separate FGs: historical + forecast) |
| **Model Registry** | Hopsworks Model Registry |
| **Primary Keys** | Time-only (no lat/lon in PKs) |
| **Automation** | GitHub Actions (hourly + daily) |
| **APIs** | Open-Meteo (Weather + AQI forecasts, with retry logic) |
| **Prediction Arch** | Historical Lags + Forecast Weather (no leakage) |
| **Explainability** | SHAP |
| **Deployment** | Docker-ready, Render + Streamlit Cloud |

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Git
- Hopsworks account (free tier available)
- Open-Meteo API access (free)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/AQI-Predictor.git
cd AQI-Predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:

```env
# Hopsworks Configuration
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name

# Location (Hyderabad, Sindh, Pakistan)
# Note: Used for API calls only; feature PKs use time only (no lat/lon)
TIMEZONE=Asia/Karachi

# Backend API
API_BASE_URL=http://localhost:8000/api
MODEL_SELECTION_METRIC=val_rmse
MODEL_SELECTION_SORT=min
```

### 5. Setup Hopsworks
1. Create account at [Hopsworks.ai](https://www.hopsworks.ai/)
2. Create new project
3. Generate API keyw
4. Add to `.env` file

---

## 🚀 Usage

### rc
python backfill/backfill_historical_data.py --start-date 2024-01-01 --end-date 2026-01-31 --batch-days 90
```
**What it does:**
- Fetches 2 years of historical data in 90-day batches
- Includes retry logic (3 attempts, exponential backoff)
- Merges weather + AQI by time, deduplicates, sorts
- Generates engineered features (lags, cyclical, interactions)
- Uploads to `aqi_historical_features` FG (PK: `time`)
- Saves local CSV backup in `data/` folder
- **Run this ONCE before starting automated pipelines**
- Typical runtime: 5-15 minutes for 2+ years
- Saves local CSV backup in data/ folder
- **Run this ONCE before starting automated pipelines**

### Stefeatures/feature_pipeline.py
```
**What it does (two independent streams):**

**Historical Features (Training)**
- Fetches last 26 hours of **observed** weather + AQI
- Processes all 26 hours to compute lag features (~24h lookback)
- Deduplicates by `time`, enforces chronological order
- Inserts only last 2 hours of **new timestamps** (~2-10 rows)
- Updates `aqi_historical_features` feature group (PK: `time`)
- **Zero forecast contamination**

**Forecast Weather Features (Inference)**
- Fetches next 72 hours of weather forecasts only (no AQI)
- Computes time-based + cyclical features (hour, day, interactions)
- Updates `weather_forecast_features` feature group (PK: `time`)
- Used at prediction time to augment historical lags

**Features:**
- Retry logic: 3 attempts with exponential backoff
- Takes ~2-3 seconds total
- Runs automatically hourly at :42 via GitHub Actions

### Step 3: Training Pipeline (Train Models)
```bash
cd src
python train/training_pipeline.py
```
**What it does:**
- Fetches 2+ years from `aqi_historical_features` only
- Sorts by `time` (enforces time-series integrity)
- Trains 5 ML models with time-series cross-validation
- Selects best model by validation RMSE
- **Prints metrics table to console**
- **Exports all models' metrics to `docs/model_metrics_table.csv`**
- Registers all models to Hopsworks Model Registry
- Generates 72h forecast cache
- Saves best model artifacts to `models/` RMSE
- Registers all models to Hopsworks
- Generates 72h forecast cache
- Saves artifacts to models/cache/
- Runs daily at 8:47 AM UTC via GitHub Actions

### Start Backend API
```bash
cd src/backend
python main.py
```
API will be available at `http://localhost:8000`

### Launch Dashboard
```bash
cd src/frontend
streamlit run app.py
```
Dashboard will open at `http://localhost:8501`

### Backfill Historical Data (Optional)
```bash
cd scripts
python backfill_historical_data.py --start-date 2024-01-01 --end-date 2026-01-31 --batch-days 90
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### 1. Get Current AQI
```http
GET /current-aqi?model=lightgbm
```
**Response:**
```json
{
  "aqi": 85.3,
  "category": "Moderate",
  "timestamp": "2026-02-02T10:00:00Z",
  "pm2_5": 35.2,
  "pm10": 68.5
}
```

#### 2. Get 72-Hour Forecast
```http
GET /predictions/next-3-days?model=lightgbm
```
**Response:**
```json
{
  "model_name": "lightgbm",
  "generated_at": "2026-02-02T10:00:00Z",
  "predictions": [
    {
      "timestamp": "2026-02-02T11:00:00Z",
      "predicted_aqi": 87.2,
      "aqi_category": "Moderate",
      "confidence": 0.92
    }
  ]
}
```

#### 3. Get Alerts
```http
GET /alerts?model=lightgbm
```
**Response:**
```json
{
  "alert_level": "warning",
  "current_aqi": 152.5,
  "peak_aqi_predicted": 178.3,
  "peak_time": "2026-02-03T14:00:00Z",
  "recommendation": "Limit outdoor activities",
  "affected_groups": ["Children", "Elderly", "Heart disease patients"]
}
```

#### 4. Get SHAP Explainability
```http
GET /shap-values?model=lightgbm
```

#### 5. Get Feature Importance
```http
GET /feature-importance?model=lightgbm
```

#### 6. Get Historical Data(fetch + process)
│       └── training-pipeline.yml     # Daily (train + register)
├── data/
│   └── backfill_*.csv               # Historical data snapshots
├── docs/
│   ├── model_metrics_table.csv      # All models' metrics (train/val/test)
│   ├── AQI_PREDICTOR_REPORT.md
│   └── SYSTEM_REPORT.md
├── models/
│   ├── lightgbm_model.pkl           # Best model
│   ├── feature_names.json
│   ├── metrics.json
│   └── cache/
│       ├── best_model_meta.json     # Selection metadata
│       ├── all_model_metrics.json   # 5 models' metrics
│       └── predictions_72h.json     # Latest 72h forecast
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis
├── src/
│   ├── backend/
│   │   ├── api_client.py            # Open-Meteo API (with retry)
│   │   ├── hopsworks_client.py      # Feature Store + Model Registry
│   │   ├── main.py                  # FastAPI server
│   │   ├── routes.py                # API endpoints
│   │   ├── schemas.py               # Pydantic models
│   │   └── services.py              # Prediction logic (historical + forecast)
│   ├── features/
│   │   ├── feature_engineering.py   # process_features() + process_forecast_features()
│   │   └── feature_pipeline.py      # Hourly: fetch + split into 2 FGs
│   ├── frontend/
│   │   └── app.py                   # Streamlit dashboard
│   ├── backfill/
│   │   └── backfill_historical_data.py  # One-time historical load (with retry)
│   └── train/
│       └── training_pipeline.py     # Train 5 models + export metrics CSV
├── tests/
│   └── test_*.py                    # Unit tests (TODO)
├── .env.example                     # Environment template
├── .gitignore
├── requirements.txt                 # Python dependencies
├── render.yaml                      # Render deployment configder
├── src/
│   ├── backend/
│   │   ├── api_client.py            # Open-Meteo API wrapper
│   │   ├── hopsworks_client.py      # Feature Store client
│   │   ├── main.py                  # FastAPI server
│   │   ├── routes.py                # API endpoints
│   │   ├── schemas.py               # Pydantic models
│   │   └── services.py              # Business logic
│   ├── features/
│   │   ├── feature_engineering.py   # Feature transformations
│   │   └── feature_pipeline.py      # Hourly data collection
│   ├── frontend/
│   │   └── app.py                   # Streamlit dashboard
│   └── train/
│       └── training_pipeline.py     # Model training
├── tests/
│   └── test_*.py                    # Unit tests
├── .env.example                     # Environment template
├── .gitignore
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🚢 Deployment

### Live Deployments
- **Streamlit App**: https://hyderabad-pearls-aqi-predictor.streamlit.app/
- **FastAPI Backend (Render)**: https://aqi-predictor-4r2g.onrender.com

### GitHub Actions (Automated)
Workflows are configured and run automatically:
- **Feature Pipeline**: Every hour at `:42` (42 minutes past the hour)
- **Training Pipeline**: Daily at 8:47 AM UTC

### Manual Deployment

#### Docker (Recommended)
```bash
# Build image
docker build -t aqi-predictor .

# Run backend
docker run -p 8000:8000 --env-file .env aqi-predictor

# Run frontend
docker run -p 8501:8501 --env-file .env aqi-predictor streamlit run src/frontend/app.py
```

#### Render (Backend)
- Uses [render.yaml](render.yaml) for one-click deploy.
- **Build command**: `pip insta

**Note:** Metrics exported to `docs/model_metrics_table.csv` (generated after each training run)ll -r requirements.txt`
- **Start command**: `cd src/backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health check**: `/health`
- **Required env vars**: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `LATITUDE`, `LONGITUDE`, `TIMEZONE`

#### Streamlit Cloud (Frontend)
- **Main file**: `src/frontend/app.py`
- **Secrets** (Settings → Secrets):
  - `API_BASE_URL = "https://aqi-predictor-4r2g.onrender.com/api"`
- Reboot the app after updating secrets.

---

## 📊 Model Performance

### Best Model: LightGBM
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **RMSE** | 2.19 | **3.24** | 3.47 |
| **MAE** | 1.64 | 2.34 | 2.51 |
| **R²** | 0.99 | 0.98 | 0.98 |

### Model Comparison (Validation RMSE)
1. 🥇 **LightGBM**: 3.24
2. 🥈 **XGBoost**: 4.18
3. 🥉 **Random Forest**: 4.46
4. **TensorFlow NN**: 6.34
5. **ElasticNet**: 6.94

### All Models Performance

| Model | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² |
|-------|-----------|----------|-----------|----------|--------|---------|
| **LightGBM** | 2.19 | 3.24 | 3.47 | 0.99 | 0.98 | 0.98 |
| **XGBoost** | 3.83 | 4.18 | 4.31 | 0.98 | 0.97 | 0.97 |
| **Random Forest** | 3.46 | 4.46 | 4.80 | 0.98 | 0.97 | 0.96 |
| **TensorFlow NN** | 5.96 | 6.34 | 6.15 | 0.95 | 0.94 | 0.94 |
| **ElasticNet** | 6.80 | 6.94 | 6.59 | 0.93 | 0.93 | 0.93 |

### Key Features (SHAP Importance)
1. PM2.5 Levels (35%)
2. PM10 Levels (18%)
3. Temperature (12%)
4. Humidity (9%)
5. Wind Speed (8%)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

---

## 🙏 Acknowledgments

- **Open-Meteo** - Free weather and air quality API
- **Hopsworks** - Feature Store and Model Registry
- **Streamlit** - Interactive dashboard framework
- **SHAP** - Model explainability library

---

## 📧 Contact

**Project Maintainer:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [@SyedaAnoosha](https://github.com/SyedaAnoosha)

---

## 🌟 Star History

If you find this project helpful, please give it a ⭐!

---

<p align="center">
  <strong>Built with ❤️ for cleaner air and healthier communities</strong>
</p>



