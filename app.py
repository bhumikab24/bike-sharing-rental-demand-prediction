import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Load the model
@st.cache_resource
def load_model():
    return joblib.load('bike_rental_model.pkl')

model = load_model()

# 2. Page Configuration
st.set_page_config(page_title="Bike Sharing Analytics", page_icon="🚲", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔧 Configuration")

hr = st.sidebar.slider("Hour of Day", 0, 23, 17)
mnth = st.sidebar.slider("Month", 1, 12, 6)
# YEAR INPUT REMOVED FROM HERE
weekday = st.sidebar.slider("Day of Week (0=Sun)", 0, 6, 2)
season_name = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
weathersit_name = st.sidebar.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
workingday_val = st.sidebar.radio("Working Day?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
holiday_val = st.sidebar.radio("Public Holiday?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

temp = st.sidebar.slider("Temperature (0-1)", 0.0, 1.0, 0.75)
hum = st.sidebar.slider("Humidity (0-1)", 0.0, 1.0, 0.4)
windspeed = st.sidebar.slider("Windspeed (0-1)", 0.0, 1.0, 0.1)

# --- TABS SETUP ---
tab1, tab2 = st.tabs(["🚀 Predict Demand", "📊 Data Insights"])

# --- TAB 1: PREDICTIONS ---
with tab1:
    st.title("🚲 Demand Prediction Engine")
    st.info("Adjust settings in the sidebar and click the button below.")

    if st.button("Run Prediction", use_container_width=True, type="primary"):
        
        # Feature Engineering
        is_rush_hour = 1 if (workingday_val == 1) and ((7 <= hr <= 9) or (16 <= hr <= 19)) else 0
        
        # We hardcode 'yr': 1 (representing 2012) so the model uses its best trend data
        input_dict = {
            'yr': 1, 
            'mnth': mnth, 
            'hr': hr, 
            'weekday': weekday, 
            'temp': temp, 
            'atemp': temp,
            'hum': hum, 
            'windspeed': windspeed, 
            'is_rush_hour': is_rush_hour,
            'season_springer': 1 if season_name == "springer" else 0,
            'season_summer': 1 if season_name == "summer" else 0,
            'season_winter': 1 if season_name == "winter" else 0,
            'weathersit_Heavy Rain': 1 if weathersit_name == "Heavy Rain" else 0,
            'weathersit_Light Snow': 1 if weathersit_name == "Light Snow" else 0,
            'weathersit_Mist': 1 if weathersit_name == "Mist" else 0,
            'workingday_Working Day': 1 if workingday_val == 1 else 0,
            'holiday_Yes': holiday_val,
            'temp_type_Hot': 1 if temp >= 0.6 else 0,
            'temp_type_Mild': 1 if 0.3 <= temp < 0.6 else 0
        }

        model_columns = [
            'yr', 'mnth', 'hr', 'weekday', 'temp', 'atemp', 'hum', 'windspeed', 
            'is_rush_hour', 'season_springer', 'season_summer', 'season_winter', 
            'weathersit_Heavy Rain', 'weathersit_Light Snow', 'weathersit_Mist', 
            'workingday_Working Day', 'holiday_Yes', 'temp_type_Hot', 'temp_type_Mild'
        ]

        df_input = pd.DataFrame([input_dict]).reindex(columns=model_columns, fill_value=0)
        prediction = int(np.round(model.predict(df_input)[0]))

        st.markdown("---")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Count", f"{max(0, prediction)} Bikes")
        col_b.metric("Time Context", "Rush Hour" if is_rush_hour else "Off-Peak")
        col_c.metric("Temp Type", "Hot" if temp >= 0.6 else ("Mild" if temp >= 0.3 else "Cold"))

        if prediction > 400:
            st.success(f"### 📈 High Demand Expected!")
        elif prediction > 100:
            st.info(f"### 📊 Moderate Demand Expected.")
        else:
            st.warning(f"### 📉 Low Demand Expected.")

# --- TAB 2: DATA INSIGHTS ---
with tab2:
    st.title("📊 Historical Patterns")
    
    # Chart 1: Seasonal Demand
    season_data = pd.DataFrame({
        'Season': ['Springer', 'Summer', 'Fall', 'Winter'],
        'Average_Count': [111, 208, 236, 198] 
    })
    fig_season = px.bar(season_data, x='Season', y='Average_Count', color='Season',
                        title="Average Bike Rentals by Season")
    st.plotly_chart(fig_season, use_container_width=True)

    # Chart 2: Hourly Trend
    hour_trend = pd.DataFrame({
        'Hour': list(range(24)),
        'Count': [40, 20, 10, 5, 10, 30, 120, 350, 500, 300, 200, 250, 300, 320, 310, 350, 550, 600, 500, 400, 300, 200, 150, 100]
    })
    fig_hour = px.line(hour_trend, x='Hour', y='Count', title="Average Hourly Rental Pattern", markers=True)
    st.plotly_chart(fig_hour, use_container_width=True)