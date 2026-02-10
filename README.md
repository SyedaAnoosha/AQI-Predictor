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

**Live App**: https://ten-pearls-aqi-predictor.streamlit.app/

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
   - **Script**: `scripts/backfill_historical_data.py`
   - Fetches 2 years of historical data (weather + AQI)
   - Creates initial training dataset (~17,520 records)
   - Run manually once before first training

**1. Feature Pipeline** (Hourly at :42 past the hour)
   - **Purpose**: Incremental updates with observed data only
   - Fetches last 26 hours (covers 24h max lag + 2h new data)
   - Processes features for all 26 hours (ensures lag values exist)
   - Filters and inserts only last 2 hours of new timestamps
   - Appends ~2-10 new rows to Hopsworks Feature Store
   - Takes ~2-3 seconds per run
   - **No forecast data** - only observed/historical AQI

2. **Training Pipeline** (Daily @ 8:47 AM UTC)
   - Load 2+ years of historical features from Hopsworks
   - Train 5 ML models with time-series cross-validation
   - Select best model (lowest validation RMSE)
   - Register all models to Hopsworks Model Registry
   - Generate 72h forecast cache
   - Save artifacts to models/cache/

3. **Inference** (On-Demand)
   - Load best model from registry
   - Generate predictions
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
| **Feature Store** | Hopsworks |
| **Model Registry** | Hopsworks Model Registry |
| **Automation** | GitHub Actions |
| **APIs** | Open-Meteo (Weather & AQI) |
| **Explainability** | SHAP |
| **Deployment** | Docker-ready, Cloud-agnostic |

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
LATITUDE=25.3792
LONGITUDE=68.3683
TIMEZONE=Asia/Karachi

# Backend API
API_BASE_URL=http://localhost:8000/api
```

### 5. Setup Hopsworks
1. Create account at [Hopsworks.ai](https://www.hopsworks.ai/)
2. Create new project
3. Generate API key
4. Add to `.env` file

---

## 🚀 Usage

### Step 1: Historical Backfill (First Time Only)
```bash
cd scripts
python backfill_historical_data.py --start-date 2024-01-01 --end-date 2026-01-31
```
**What it does:**
- Fetches 2 years of historical data in 90-day batches
- Generates engineered features
- Uploads to Hopsworks Feature Store
- Saves local CSV backup in data/ folder
- **Run this ONCE before starting automated pipelines**

### Step 2: Incremental Feature Pipeline
```bash
cd src
python -m features.feature_pipeline
```
**What it does:**
- Fetches last 26 hours of observed data (for lag feature context)
- Processes all data to compute 24h lag features correctly
- Inserts only last 2 hours of new timestamps (~2-10 rows)
- Uses only observed AQI (no forecast contamination)
- Takes 2-3 seconds
- Runs automatically hourly at :42 via GitHub Actions

### Step 3: Training Pipeline (Train Models)
```bash
cd src
python -m train.training_pipeline
```
**What it does:**
- Trains 5 ML models on 2+ years of data
- Selects best model by validation RMSE
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

#### 6. Get Historical Data
```http
GET /historical?days=30
```

#### 7. Get Model Metrics
```http
GET /model-metrics/all
```

**Full API docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 📂 Project Structure

```
AQI-Predictor/
├── .github/
│   └── workflows/
│       ├── feature-pipeline.yml      # Hourly data collection
│       └── training-pipeline.yml     # Daily model training
├── data/
│   └── backfill_*.csv               # Historical data exports
├── models/                          # placeholder
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis
├── scripts/
│   └── backfill_historical_data.py  # Historical data loader
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
- **Streamlit App**: https://ten-pearls-aqi-predictor.streamlit.app/
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

#### Cloud Platforms
- **Render (Backend)**
  - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
  - Set environment variables: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `LATITUDE`, `LONGITUDE`, `TIMEZONE`
  - Runtime: Python 3.11 (`runtime.txt`)
- **Streamlit Cloud (Frontend)**
  - Main file: `src/frontend/app.py`
  - Add secrets in Streamlit Cloud:
    - `API_BASE_URL = "https://aqi-predictor-4r2g.onrender.com/api"`
  - Reboot app after updating secrets

Other platforms:
- **AWS**: Deploy on EC2, ECS, or Lambda
- **GCP**: Cloud Run or App Engine
- **Azure**: App Service or Container Instances
- **Heroku**: Procfile included

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
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 🌟 Star History

If you find this project helpful, please give it a ⭐!

---

<p align="center">
  <strong>Built with ❤️ for cleaner air and healthier communities</strong>
</p>


