# Pearls AQI Predictor

## Project Overview

Predict the Air Quality Index (AQI) in Hyderabad, Sindh for the next 3 days using a fully serverless setup.  

This project covers an end-to-end machine learning workflow, including automated data collection, feature processing, model training, and real-time predictions shown in a web dashboard.

## Technology Stack

**Languages, tools, and platforms:**

- Python  
- Scikit-learn  
- TensorFlow  
- Hopsworks 
- GitHub Actions
- Streamlit  
- Flask
- OpenWeatherMap's Air Pollution API and Open-Meteo's Historical Weather API
- SHAP  
- Git  

---

## Key Features

### Feature Pipeline

- Fetch raw weather and pollutant data from Open-Meteo's Historical Weather API and OpenWeatherMap's Air Pollution API respectively.
- Generate features from raw data, including:
  - Time-based features like hour, day, and month  
  - Derived values such as AQI change rate  
- Store processed features in a Feature Store using Hopsworks  

---

### Historical Data Backfill

- Run the feature pipeline on past dates  
- Generate training datasets from historical data  
- Build a complete dataset for model training and evaluation  

---

### Model Training Pipeline

- Load historical features and target values from the Feature Store  
- Train and compare multiple models, including:
  - Random Forest  
  - Ridge Regression  
  - TensorFlow or PyTorch models  
- Measure performance using RMSE, MAE, and R²  
- Save trained models to a Model Registry  

---

### Automated CI/CD

- Run the feature pipeline automatically every hour  
- Retrain models daily to keep predictions fresh  
- Use GitHub Actions  

---

### Web Application Dashboard

- Load models and features directly from the Feature Store  
- Computes model predictions and shows them on a simple and descriptive dashboard.
- Generate real-time AQI predictions for the next 3 days  
- Show results in an interactive dashboard using Streamlit with FastAPI  

---

### Data Insights and Explainability

- Perform Exploratory Data Analysis to spot trends and patterns  
- Use SHAP to explain model predictions  
- Set up alerts for unhealthy or hazardous AQI levels  
- Support multiple forecasting approaches, from classical methods to neural network.

---
