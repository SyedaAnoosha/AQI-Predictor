import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set
    page_title="Hyderabad AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    body, div, section, span, label, input, textarea, select, button {
        font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', 'Manrope', sans-serif;
        letter-spacing: 0.01em;
    }

    code, pre, .code, .stCode, .stMarkdown code {
        font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
    }

    :root {
        --primary-color: #1f77b4;
        --good-color: #00cc00;
        --moderate-color: #ffcc00;
        --unhealthy-color: #ff6600;
        --hazard-color: #ff0000;
        --very-unhealthy-color: #9933cc;
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.1), rgba(31, 119, 180, 0.05));
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .good { color: #00cc00; font-weight: bold; }
    .moderate { color: #ffcc00; font-weight: bold; }
    .unhealthy { color: #ff6600; font-weight: bold; }
    .very-unhealthy { color: #9933cc; font-weight: bold; }
    .hazardous { color: #ff0000; font-weight: bold; }
    
    .aqi-category-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white !important;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .aqi-category-box div {
        color: white !important;
    }
    
    .aqi-box-good { 
        background: linear-gradient(135deg, #00cc00, #00aa00) !important;
    }
    .aqi-box-moderate { 
        background: linear-gradient(135deg, #ffcc00, #ff9900) !important;
        color: #000000 !important;
    }
    .aqi-box-moderate div {
        color: #000000 !important;
    }
    .aqi-box-unhealthy-for-sensitive-groups { 
        background: linear-gradient(135deg, #ff8c00, #ff6600) !important;
    }
    .aqi-box-unhealthy { 
        background: linear-gradient(135deg, #ff6600, #ff3300) !important;
    }
    .aqi-box-very-unhealthy { 
        background: linear-gradient(135deg, #9933cc, #7700aa) !important;
    }
    .aqi-box-hazardous { 
        background: linear-gradient(135deg, #ff0000, #cc0000) !important;
    }
    
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .header-title h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .header-title p {
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    .info-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .recommendation-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .recommendation-good { 
        background-color: rgba(0, 204, 0, 0.1);
        border-left-color: #00cc00;
    }
    
    .recommendation-caution {
        background-color: rgba(255, 204, 0, 0.1);
        border-left-color: #ffcc00;
    }
    
    .recommendation-warning {
        background-color: rgba(255, 102, 0, 0.1);
        border-left-color: #ff6600;
    }
    
    .recommendation-alert {
        background-color: rgba(255, 0, 0, 0.1);
        border-left-color: #ff0000;
    }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .chart-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

def get_aqi_color(aqi: float) -> str:
    if aqi <= 50:
        return "#00cc00"
    elif aqi <= 100:
        return "#ffcc00"
    elif aqi <= 150:
        return "#ff6600"
    elif aqi <= 200:
        return "#ff0000"
    elif aqi <= 300:
        return "#9933cc"
    else:
        return "#660000"

def get_aqi_category_text(aqi: float) -> str:
    if aqi <= 50:
        return "🟢 Good"
    elif aqi <= 100:
        return "🟡 Moderate"
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "🔴 Unhealthy"
    elif aqi <= 300:
        return "🟣 Very Unhealthy"
    else:
        return "🟤 Hazardous"

def get_aqi_category_name(aqi: float) -> str:
    if aqi <= 50:
        return "good"
    elif aqi <= 100:
        return "moderate"
    elif aqi <= 150:
        return "unhealthy-for-sensitive-groups"
    elif aqi <= 200:
        return "unhealthy"
    elif aqi <= 300:
        return "very-unhealthy"
    else:
        return "hazardous"

def get_readable_feature_name(feature: str) -> str:
    mapping = {
        'pm2_5': 'PM2.5 Levels',
        'pm10': 'PM10 Levels',
        'temperature_2m': 'Temperature',
        'relative_humidity_2m': 'Humidity',
        'wind_speed_10m': 'Wind Speed',
        'wind_direction_10m': 'Wind Direction',
        'precipitation': 'Rainfall',
        'surface_pressure': 'Air Pressure',
        'cloud_cover': 'Cloud Cover',
        'hour': 'Time of Day',
        'day_of_week': 'Day of Week',
        'month': 'Season',
        'is_weekend': 'Weekend Effect',
        'aqi_lag_': 'Past AQI ',
        'aqi_change_rate': 'AQI Trend',
        'temp_pm_interaction': 'Temperature-PM Interaction',
        'wind_pm_interaction': 'Wind-PM Interaction',
    }
    
    for key, readable in mapping.items():
        if key in feature.lower():
            return readable
    
    return feature.replace('_', ' ').title()

def generate_shap_summary(top_features: list) -> str:
    if not top_features or len(top_features) < 3:
        return "Insufficient data for explanation."
    
    top_3 = top_features[:3]
    feature_names = [get_readable_feature_name(f['feature_name']) for f in top_3]
    
    pollutants = [f for f in feature_names if any(p in f for p in ['PM2.5', 'PM10', 'AQI'])]
    weather = [f for f in feature_names if any(w in f for w in ['Wind', 'Temperature', 'Humidity', 'Rain'])]
    
    summary_parts = []
    
    if pollutants:
        summary_parts.append(f"{', '.join(pollutants[:2])}")
    if weather:
        summary_parts.append(f"{', '.join(weather[:2])}")
    
    if len(summary_parts) == 2:
        return f"Today's AQI is primarily driven by {summary_parts[0]} with {summary_parts[1]} playing a secondary role."
    elif len(summary_parts) == 1:
        return f"Today's AQI is primarily influenced by {summary_parts[0]}."
    else:
        return f"The top factors affecting AQI are {', '.join(feature_names[:2])}."

def get_health_recommendations(aqi: float) -> dict:
    if aqi <= 50:
        return {
            "general": "Air quality is satisfactory. Enjoy outdoor activities!",
            "sensitive": "No restrictions. Safe for all outdoor activities.",
            "activity": "All outdoor activities are recommended.",
            "precautions": "No precautions needed."
        }
    elif aqi <= 100:
        return {
            "general": "Air quality is acceptable. Slight risk for sensitive groups.",
            "sensitive": "Sensitive individuals should consider limiting prolonged outdoor activities.",
            "activity": "Outdoor activities are generally safe.",
            "precautions": "Sensitive groups: monitor air quality and symptoms."
        }
    elif aqi <= 150:
        return {
            "general": "Unhealthy for sensitive groups.",
            "sensitive": "Sensitive individuals should limit outdoor activities.",
            "activity": "Outdoor activities should be limited for sensitive groups.",
            "precautions": "N95 masks recommended for sensitive groups."
        }
    elif aqi <= 200:
        return {
            "general": "Everyone may experience health effects.",
            "sensitive": "Avoid outdoor activities. Stay indoors.",
            "activity": "Everyone should limit outdoor activities.",
            "precautions": "Wear N95 masks. Use air purifiers indoors."
        }
    elif aqi <= 300:
        return {
            "general": "Very unhealthy. Health warnings.",
            "sensitive": "Remain indoors. Use air purifiers.",
            "activity": "Avoid outdoor activities completely.",
            "precautions": "Stay indoors. Close windows and doors. Use air purifiers."
        }
    else:
        return {
            "general": "Hazardous air quality. Health emergency.",
            "sensitive": "Remain indoors. Seek medical help if symptoms.",
            "activity": "No outdoor activities permitted.",
            "precautions": "Emergency level. Everyone should remain indoors."
        }

MODEL_DESCRIPTIONS = {
    "lightgbm": {
        "label": "LightGBM",
        "algorithm": "Gradient boosting decision trees tuned for speed and accuracy",
        "strengths": "Handles non-linear relationships and interactions well with efficient training",
    },
    "xgboost": {
        "label": "XGBoost",
        "algorithm": "Extreme Gradient Boosting with robust regularization",
        "strengths": "Strong tabular performance, good handling of missing values",
    },
    "random_forest": {
        "label": "Random Forest",
        "algorithm": "Ensemble of decorrelated decision trees",
        "strengths": "Stable baseline, resistant to overfitting, interpretable feature splits",
    },
    "elasticnet": {
        "label": "Elastic Net",
        "algorithm": "Linear model with L1/L2 regularization",
        "strengths": "Good for linear relationships; balances sparsity and stability",
    },
    "tensorflow_nn": {
        "label": "TensorFlow NN",
        "algorithm": "Feed-forward neural network",
        "strengths": "Can capture complex non-linear patterns when enough data is available",
    },
}

def get_model_info(model_key: str) -> dict:
    key = (model_key or "").lower()
    return MODEL_DESCRIPTIONS.get(key, {
        "label": model_key or "Model",
        "algorithm": "Machine learning model",
        "strengths": "Trained on historical AQI and weather with engineered features",
    })


def fmt_metric_value(value: float) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return ""

def pick_metric(metrics: dict, candidates: list) -> str:
    for key in candidates:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return fmt_metric_value(metrics[key])
    return ""

def get_selected_model() -> str:
    return st.session_state.get("selected_model") or st.session_state.get("default_model", "lightgbm")

@st.cache_data(ttl=600, show_spinner=False)
def fetch_models_list():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"available_models": ["lightgbm"], "default_model": "lightgbm", "loaded_models": []}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_model_metrics():
    try:
        response = requests.get(f"{API_BASE_URL}/model-metrics/all")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def load_local_metrics_cache() -> dict:
    try:
        root_dir = Path(__file__).resolve().parents[2]
        cache_path = root_dir / "models" / "cache" / "all_model_metrics.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                normalized = {}
                for name, metrics in raw.items():
                    if not isinstance(metrics, dict):
                        continue
                    key = name.strip().lower().replace(" ", "_")
                    normalized[key] = metrics
                return normalized
    except Exception:
        return {}
    return {}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_forecast(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/next-3-days", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_current_aqi(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/current-aqi", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_alerts(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/alerts", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(days: int = 30):
    try:
        response = requests.get(f"{API_BASE_URL}/historical?days={days}")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_feature_importance(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/feature-importance", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_shap_values(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/shap-values", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_health_guide(model_name: str):
    try:
        response = requests.get(f"{API_BASE_URL}/health-guide", params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def render_home_page(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>🌍 AQI Predictor Dashboard</h1>
            <p>Real-time Air Quality Index Predictions for Hyderabad, Sindh</p>
        </div>
    """, unsafe_allow_html=True)
    
    current_aqi = fetch_current_aqi(selected_model)
    forecast = fetch_forecast(selected_model)
    alerts = fetch_alerts(selected_model)
    
    st.subheader("📊 Current Air Quality Status")
    
    if current_aqi:
        col_spacer1, col_aqi, col_spacer2 = st.columns([1, 2, 1])
        
        with col_aqi:
            aqi_value = current_aqi.get('aqi', 100)
            category_name = get_aqi_category_name(aqi_value)
            st.markdown(f"""
                <div class="aqi-category-box aqi-box-{category_name}">
                    <div style="font-size: 36px; margin-bottom: 10px;">{aqi_value:.0f}</div>
                    <div>Current AQI</div>
                </div>
            """, unsafe_allow_html=True)
            st.caption(get_aqi_category_text(aqi_value))
    else:
        st.error("❌ Unable to fetch current air quality data!")
      
    st.divider()
    
    if alerts and alerts.get('alert_level') and alerts.get('alert_level').lower() != 'normal':
        st.markdown("### ⚠️ Active Alerts")
        
        alert_level = alerts.get('alert_level', 'NORMAL').upper()
        
        if alert_level == 'HAZARD':
            st.error(f"🚨 **HAZARD LEVEL ALERT** - Health Risk to General Population")
        elif alert_level == 'WARNING':
            st.warning(f"⚠️ **WARNING LEVEL ALERT** - Potential Health Effects")
        else:
            st.info(f"ℹ️ **{alert_level} ALERT** - Monitor Air Quality")
        
        alert_cols = st.columns(3)
        with alert_cols[0]:
            st.metric("Peak AQI (72h)", f"{alerts.get('peak_aqi_predicted', 100):.0f}")
        with alert_cols[1]:
            peak_time = alerts.get('peak_time', datetime.now())
            if isinstance(peak_time, str):
                peak_time = pd.to_datetime(peak_time)
            st.metric("Peak Time", peak_time.strftime('%a %I:%M %p'))
        with alert_cols[2]:
            st.metric("Duration", "Next 72 hours")
    
    st.divider()
    
    st.subheader("📈 72-Hour AQI Forecast")
    
    if forecast and 'predictions' in forecast:
        predictions = forecast['predictions']
        df_forecast = pd.DataFrame({
            'Time': [p['timestamp'] for p in predictions],
            'AQI': [p['predicted_aqi'] for p in predictions],
            'Category': [p['aqi_category'] for p in predictions]
        })
        
        df_forecast['Time'] = pd.to_datetime(df_forecast['Time'])
        
        fig = go.Figure()
        
        colors = [get_aqi_color(aqi) for aqi in df_forecast['AQI']]
        
        fig.add_trace(go.Scatter(
            x=df_forecast['Time'],
            y=df_forecast['AQI'],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(color='#1f77b4', width=3),
            marker=dict(
                size=2,
                color=colors,
                line=dict(width=2, color='white')
            ),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good (50)")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate (100)")
        fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy (150)")
        
        fig.update_layout(
            title="AQI Forecast (Next 72 Hours)",
            xaxis_title="Time",
            yaxis_title="AQI Value",
            hovermode='x unified',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_pred = max(predictions, key=lambda x: x['predicted_aqi'])
            st.metric("Peak AQI", f"{peak_pred['predicted_aqi']:.0f}", 
                     get_aqi_category_text(peak_pred['predicted_aqi']))
        
        with col2:
            low_pred = min(predictions, key=lambda x: x['predicted_aqi'])
            st.metric("Lowest AQI", f"{low_pred['predicted_aqi']:.0f}",
                     get_aqi_category_text(low_pred['predicted_aqi']))
        
        with col3:
            avg_aqi = sum(p['predicted_aqi'] for p in predictions) / len(predictions)
            st.metric("Average AQI", f"{avg_aqi:.0f}",
                     get_aqi_category_text(avg_aqi))
    
    st.divider()
    
    st.subheader("📉 Recent AQI Trend (Last 3 Days)")
    recent_hist = fetch_historical_data(3)
    if recent_hist and 'data_points' in recent_hist and recent_hist['data_points']:
        df_recent = pd.DataFrame(recent_hist['data_points'])
        if 'date' in df_recent.columns and 'avg_aqi' in df_recent.columns:
            df_recent['date'] = pd.to_datetime(df_recent['date'])
            
            fig_recent = go.Figure()
            fig_recent.add_trace(go.Scatter(
                x=df_recent['date'],
                y=df_recent['avg_aqi'],
                mode='lines+markers',
                name='Daily Avg AQI',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6, color=[get_aqi_color(aqi) for aqi in df_recent['avg_aqi']])
            ))
            
            fig_recent.update_layout(
                xaxis_title='Date',
                yaxis_title='Average AQI',
                height=250,
                template='plotly_white',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_recent, width='stretch')
        else:
            st.info("Recent trend data unavailable")
    
    st.divider()
    
    st.subheader("🌤️ Current Weather & Pollution Context")
    if current_aqi:
        with st.container():
            st.markdown("""
                <style>
                .scrollable-metrics {
                    max-height: 200px;
                    overflow-x: auto;
                    overflow-y: hidden;
                    padding: 10px 0;
                }
                </style>
            """, unsafe_allow_html=True)
            
            col_w1, col_w2, col_w3, col_w4, col_w5, col_w6 = st.columns(6)
            
            with col_w1:
                pm25 = current_aqi.get('pm2_5', 0)
                st.metric("💨 PM2.5", f"{pm25:.1f} μg/m³")
            
            with col_w2:
                pm10 = current_aqi.get('pm10', 0)
                st.metric("💨 PM10", f"{pm10:.1f} μg/m³")
            
            with col_w3:
                temp = current_aqi.get('temperature', 0)
                st.metric("🌡️ Temperature", f"{temp:.1f}°C")
            
            with col_w4:
                humidity = current_aqi.get('humidity', 0)
                st.metric("💧 Humidity", f"{humidity:.0f}%")
            
            with col_w5:
                wind_speed = current_aqi.get('wind_speed', 0)
                st.metric("🌬️ Wind Speed", f"{wind_speed:.1f} km/h")
            
            with col_w6:
                precip = current_aqi.get('precipitation', 0)
                rain_status = "Yes ☔" if precip > 0.1 else "No"
                st.metric("🌧️ Rain", rain_status)
        
        st.caption("💡 Note: Rain and wind help disperse pollutants, improving air quality")
    
    st.divider()
    
    st.subheader("💡 Quick Tips & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🏃 Outdoor Activities:**
        - Green/Yellow AQI: Safe for most activities
        - Orange/Red AQI: Limit outdoor exposure
        - Purple/Maroon: Avoid outdoor activities
        """)
    
    with col2:
        st.info("""
        **😷 Health Precautions:**
        - Wear N95 mask in Orange/Red levels
        - Use air purifiers indoors
        - Keep windows closed during alerts
        - Monitor sensitive group members
        """)

def render_detailed_forecast(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>📊 Detailed Forecast</h1>
            <p>Hour-by-hour AQI predictions with confidence scores</p>
        </div>
    """, unsafe_allow_html=True)
    
    forecast = fetch_forecast(selected_model)
    if forecast and 'predictions' in forecast:
        predictions = forecast['predictions']
        
        df_detailed = pd.DataFrame({
            'Time': [pd.to_datetime(p['timestamp']).strftime('%a %m/%d %I:%M %p') for p in predictions],
            'Predicted AQI': [f"{p['predicted_aqi']:.0f}" for p in predictions],
            'Category': [p['aqi_category'] for p in predictions],
            'Confidence': [f"{p.get('confidence', 0.85):.1%}" for p in predictions]
        })
        
        st.subheader("📋 Hourly Predictions Table")
        st.dataframe(df_detailed, width='stretch')
        
        csv = df_detailed.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"aqi_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.divider()
        
        st.subheader("📅 Daily Summary")
        
        daily_data = {}
        for pred in predictions:
            date = pd.to_datetime(pred['timestamp']).date()
            if date not in daily_data:
                daily_data[date] = []
            daily_data[date].append(pred['predicted_aqi'])
        
        summary_data = []
        for date, aqi_values in sorted(daily_data.items()):
            summary_data.append({
                'Date': date.strftime('%A, %B %d'),
                'Min AQI': f"{min(aqi_values):.0f}",
                'Max AQI': f"{max(aqi_values):.0f}",
                'Avg AQI': f"{sum(aqi_values)/len(aqi_values):.0f}",
                'Hours': len(aqi_values)
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, width='stretch')
        
        st.divider()
        
        st.subheader("📈 Day-by-Day Breakdown")
        
        for date, aqi_values in sorted(daily_data.items()):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{date.strftime('%A')}", f"{max(aqi_values):.0f} (Peak)")
            with col2:
                avg = sum(aqi_values)/len(aqi_values)
                st.metric("Average", f"{avg:.0f}", get_aqi_category_text(avg))
            with col3:
                st.metric("Minimum", f"{min(aqi_values):.0f}")

def render_health_guidance(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>❤️ Health Guidance & Recommendations</h1>
            <p>Personalized health recommendations based on current AQI</p>
        </div>
    """, unsafe_allow_html=True)
    
    current_aqi_data = fetch_current_aqi(selected_model)
    alerts = fetch_alerts(selected_model)
    
    if current_aqi_data:
        current_aqi = current_aqi_data.get('aqi', 100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            category_name = get_aqi_category_name(current_aqi)
            st.markdown(f"""
                <div class="aqi-category-box aqi-box-{category_name}">
                    <div style="font-size: 32px; margin-bottom: 10px;">{current_aqi:.0f}</div>
                    <div>Current AQI Status</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if alerts and alerts.get('alert_level') and alerts.get('alert_level').lower() != 'normal':
                alert_level = alerts.get('alert_level').upper()
                st.error(f"⚠️ **Alert Level:** {alert_level}")
            else:
                st.success("✅ No Active Alerts")
        
        st.divider()
        
        recommendations = get_health_recommendations(current_aqi)
        
        st.subheader("📋 Health Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="recommendation-box recommendation-{'good' if current_aqi <= 50 else 'caution' if current_aqi <= 100 else 'warning' if current_aqi <= 150 else 'alert'}">
                <strong>👥 General Public:</strong><br>
                {recommendations['general']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="recommendation-box recommendation-{'good' if current_aqi <= 50 else 'caution' if current_aqi <= 100 else 'warning' if current_aqi <= 150 else 'alert'}">
                <strong>🏃 Activity Recommendation:</strong><br>
                {recommendations['activity']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="recommendation-box recommendation-{'good' if current_aqi <= 50 else 'caution' if current_aqi <= 100 else 'warning' if current_aqi <= 150 else 'alert'}">
                <strong>⚕️ Sensitive Groups:</strong><br>
                {recommendations['sensitive']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="recommendation-box recommendation-{'good' if current_aqi <= 50 else 'caution' if current_aqi <= 100 else 'warning' if current_aqi <= 150 else 'alert'}">
                <strong>🛡️ Health Precautions:</strong><br>
                {recommendations['precautions']}
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("📖 AQI Scale Explanation")
        
        aqi_scale = [
            ("0-50", "🟢 Good", "Air quality is satisfactory", "Safe for all activities"),
            ("51-100", "🟡 Moderate", "Acceptable, minor risk", "Generally safe for most"),
            ("101-150", "🟠 Unhealthy for Sensitive", "Sensitive groups affected", "Restrict outdoor activity"),
            ("151-200", "🔴 Unhealthy", "General population affected", "Limit outdoor activities"),
            ("201-300", "🟣 Very Unhealthy", "Everyone affected", "Avoid outdoor activities"),
            ("301+", "🟤 Hazardous", "Health emergency", "Stay indoors")
        ]
        
        scale_df = pd.DataFrame(
            [[r[0], r[1], r[2], r[3]] for r in aqi_scale],
            columns=["AQI Range", "Category", "Health Effect", "Action Required"]
        )
        st.dataframe(scale_df, width='stretch')

def render_historical_data(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>📈 Historical Data & Trends</h1>
            <p>View past AQI trends and patterns</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days = st.slider("Select time period (days)", 7, 365, 30)
    
    with col2:
        st.write("")
        st.write("")
        if st.button("📊 Load Historical Data", width='stretch'):
            st.rerun()
    
    with col3:
        st.write("")
        st.write("")
        st.caption(f"Showing last {days} days")
    
    historical = fetch_historical_data(days)
    
    if historical:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average AQI", f"{historical.get('average_aqi', 100):.0f}")
        
        with col2:
            st.metric("Min AQI", f"{historical.get('min_aqi', 50):.0f}")
        
        with col3:
            st.metric("Max AQI", f"{historical.get('max_aqi', 150):.0f}")
        
        with col4:
            aqi_range = historical.get('max_aqi', 150) - historical.get('min_aqi', 50)
            st.metric("AQI Range", f"{aqi_range:.0f}")
        
        st.divider()
        
        st.subheader(f"Daily Average AQI - Last {days} Days")

        if 'data_points' in historical and historical['data_points']:
            df_trend = pd.DataFrame(historical['data_points'])

            if 'date' in df_trend.columns and 'avg_aqi' in df_trend.columns:
                df_trend['date'] = pd.to_datetime(df_trend['date'])

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_trend['date'],
                    y=df_trend['avg_aqi'],
                    mode='lines+markers',
                    name='Daily Avg AQI',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8, color=[get_aqi_color(aqi) for aqi in df_trend['avg_aqi']], line=dict(width=1, color='white'))
                ))

                fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good (50)")
                fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate (100)")
                fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy (150)")

                fig.update_layout(
                    title=f'Daily Average AQI ({len(df_trend)} days)',
                    xaxis_title='Date',
                    yaxis_title='Average AQI',
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )

                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Historical data is missing daily aggregation fields")
        else:
            st.info("No historical data points available")


def render_alerts(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>⚠️ Alerts & Warnings</h1>
            <p>Current and upcoming air quality alerts</p>
        </div>
    """, unsafe_allow_html=True)
    
    alerts = fetch_alerts(selected_model)
    current_aqi_data = fetch_current_aqi(selected_model)
    
    if alerts:
        if alerts.get('alert_level') == 'normal' or not alerts.get('alert_level'):
            st.success("✅ **No Alerts** - Air quality is within acceptable levels")
        else:
            alert_level = alerts.get('alert_level', 'NORMAL').upper()
            current_aqi = alerts.get('current_aqi', 100)
            peak_aqi = alerts.get('peak_aqi_predicted', 100)
            
            if alert_level == 'HAZARD':
                st.error(f"🚨 **HAZARD LEVEL ALERT**")
            elif alert_level == 'WARNING':
                st.warning(f"⚠️ **WARNING LEVEL ALERT**")
            else:
                st.info(f"ℹ️ **{alert_level} LEVEL ALERT**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current AQI", f"{current_aqi:.0f}")
            
            with col2:
                st.metric("Peak AQI (72h)", f"{peak_aqi:.0f}")
            
            with col3:
                peak_time = alerts.get('peak_time', datetime.now())
                if isinstance(peak_time, str):
                    peak_time = pd.to_datetime(peak_time)
                st.metric("Peak Time", peak_time.strftime('%a %I:%M %p'))
            
            st.divider()
            
            st.subheader("🏥 Health Recommendation")
            recommendation = alerts.get('recommendation', 'Monitor air quality regularly')
            st.info(recommendation)
            
            if alerts.get('affected_groups'):
                st.subheader("👥 Affected Groups")
                affected = alerts.get('affected_groups', [])
                if isinstance(affected, list):
                    for group in affected:
                        st.warning(f"• {group}")

def render_model_info(selected_model: str):
    st.markdown("""
        <div class="header-title">
            <h1>🤖 Model Information & Explainability</h1>
            <p>Understand how the AQI predictions are made</p>
        </div>
    """, unsafe_allow_html=True)
    
    metrics_all = fetch_all_model_metrics()
    metrics_payload = metrics_all.get("metrics") if metrics_all else None
    if not metrics_payload or len(metrics_payload) <= 1:
        local_metrics = load_local_metrics_cache()
        if local_metrics:
            if metrics_all is None:
                metrics_all = {"metrics": local_metrics, "default_model": "lightgbm"}
            else:
                metrics_all["metrics"] = local_metrics
            metrics_payload = local_metrics
    
    if metrics_all and metrics_payload:
        st.subheader("📈 Model Metrics")
        rows = []
        for name, m in metrics_payload.items():
            label = get_model_info(name).get("label", name)
            rows.append({
                "Model": label,
                "Train RMSE": pick_metric(m, ["train_rmse", "rmse_train", "rmse_tr"]),
                "Val RMSE": pick_metric(m, ["val_rmse", "rmse_val", "validation_rmse"]),
                "Test RMSE": pick_metric(m, ["test_rmse", "rmse"]),
                "Train MAE": pick_metric(m, ["train_mae", "mae_train", "mae_tr"]),
                "Val MAE": pick_metric(m, ["val_mae", "mae_val", "validation_mae"]),
                "Test MAE": pick_metric(m, ["test_mae", "mae"]),
                "Train R²": pick_metric(m, ["train_r2", "r2_train", "r2_tr"]),
                "Val R²": pick_metric(m, ["val_r2", "r2_val", "validation_r2"]),
                "Test R²": pick_metric(m, ["test_r2", "r2"]),
            })
        if rows:
            df_metrics = pd.DataFrame(rows)
            st.dataframe(df_metrics, width='stretch')
    
    st.divider()
    
    st.subheader("📊 Feature Importance (SHAP)")
    st.info("🎯 **SHAP (SHapley Additive exPlanations)** - Game theory-based approach showing how each feature contributes to predictions")
    
    shap_data = fetch_shap_values(selected_model)
    
    if shap_data and 'top_features' in shap_data:
        top_features = shap_data['top_features'][:15]
        
        summary = generate_shap_summary(top_features[:5])
        st.success(f"🔍 **Key Insight:** {summary}")
        
        if 'base_value' in shap_data:
            st.caption(f"Base value (expected AQI without features): {shap_data['base_value']:.2f}")
        
        df_features = pd.DataFrame({
            'Feature': [get_readable_feature_name(f['feature_name']) for f in top_features],
            'SHAP Importance': [f['importance_score'] for f in top_features]
        })
        
        fig = px.bar(
            df_features,
            x='SHAP Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Features by SHAP Importance',
            height=500,
            color='SHAP Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.dataframe(df_features, width='stretch')
        
    else:
        st.warning("⚠️ SHAP values unavailable, showing regular feature importance")
        feature_importance = fetch_feature_importance(selected_model)
        
        if feature_importance and 'top_features' in feature_importance:
            top_features = feature_importance['top_features'][:15]
            
            summary = generate_shap_summary(top_features[:5])
            st.success(f"🔍 **Key Insight:** {summary}")
            
            df_features = pd.DataFrame({
                'Feature': [get_readable_feature_name(f['feature_name']) for f in top_features],
                'Importance': [f['importance_score'] for f in top_features]
            })
            
            fig = px.bar(
                df_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Features by Importance',
                height=500,
                color='Importance',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white'
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.dataframe(df_features, width='stretch')
        else:
            st.warning("Feature importance data unavailable")
    
    st.divider()
    
    st.subheader("ℹ️ About the Model")

    model_info = get_model_info(selected_model)

    st.markdown(f"""
        **{model_info['label']}**
        
        - **Algorithm**: {model_info['algorithm']}
        - **Strengths**: {model_info['strengths']}
        - **Training Data**: 2+ years of historical AQI and weather data
        - **Features**: Engineered signals across weather, pollutants, temporal patterns, lags, and rolling trends
        - **Forecast Horizon**: 72 hours ahead
        - **Update Frequency**: Daily model retraining
        - **Location**: Hyderabad, Sindh, Pakistan
        - **Coordinates**: 25.3548°N, 68.3711°E
        """)

def main():
    st.sidebar.title("🌍 AQI Predictor")
    st.sidebar.markdown("---")

    models_info = fetch_models_list()
    available_models = models_info.get("available_models", ["lightgbm"])
    default_model = models_info.get("default_model", available_models[0]) if available_models else "lightgbm"

    if "default_model" not in st.session_state:
        st.session_state["default_model"] = default_model
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = default_model

    selected_model = st.session_state["selected_model"]
    st.sidebar.markdown(f"**Model:** {selected_model}")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "📊 Detailed Forecast",
            "❤️ Health Guidance",
            "📈 Historical Data",
            "⚠️ Alerts",
            "🤖 Model Info"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    **AQI Predictor Dashboard**
    
    Real-time air quality predictions using machine learning.
    
    📍 Location: Hyderabad, Sindh
                    \n
    🔄 Updates: Hourly forecasts, daily model training
                    \n
    📊 Data Source: Open-Meteo API
    """)
    
    if page == "🏠 Home":
        render_home_page(selected_model)
    elif page == "📊 Detailed Forecast":
        render_detailed_forecast(selected_model)
    elif page == "❤️ Health Guidance":
        render_health_guidance(selected_model)
    elif page == "📈 Historical Data":
        render_historical_data(selected_model)
    elif page == "⚠️ Alerts":
        render_alerts(selected_model)
    elif page == "🤖 Model Info":
        render_model_info(selected_model)
    
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
            <p>AQI Predictor Dashboard | Real-time Air Quality Predictions</p>
            <p>Location: Hyderabad, Sindh (25.3548°N, 68.3711°E) | Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
 air quality predictions using machine learning.
    
    📍 Location: Hyderabad, Sindh
                    \n
    🔄 Updates: Hourly forecasts, daily model training
                    \n
    📊 Data Source: Open-Meteo API
    """)
    
    if page == "🏠 Home":
        render_home_page(selected_model)
    elif page == "📊 Detailed Forecast":
        render_detailed_forecast(selected_model)
    elif page == "❤️ Health Guidance":
        render_health_guidance(selected_model)
    elif page == "📈 Historical Data":
        render_historical_data(selected_model)
    elif page == "⚠️ Alerts":
        render_alerts(selected_model)
    elif page == "🤖 Model Info":
        render_model_info(selected_model)
    
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
            <p>AQI Predictor Dashboard | Real-time Air Quality Predictions</p>
            <p>Location: Hyderabad, Sindh (25.3548°N, 68.3711°E) | Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()