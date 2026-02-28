import streamlit as st
import pandas as pd
import requests
import os
import cv2
from datetime import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="JamSniper", layout="centered")
st.title("🚦 JamSniper: Causeway Traffic")

# --- SIDEBAR: CALIBRATION SLIDERS ---
# These match the coordinates used by the bot to draw the divider line
st.sidebar.title("📏 Line Calibration")
tx = st.sidebar.slider("Top X", 0.0, 1.0, 1.0)
ty = st.sidebar.slider("Top Y", 0.0, 1.0, 0.31)
bx = st.sidebar.slider("Bottom X", 0.0, 1.0, 0.35)
by = st.sidebar.slider("Bottom Y", 0.0, 1.0, 0.93)

# 2. WEATHER (SINGAPORE RAINFALL SENSORS)
def get_weather():
    try:
        url = "https://api.data.gov.sg/v1/environment/rainfall"
        data = requests.get(url).json()
        stations = data['metadata']['stations']
        readings = data['items'][0]['readings']
        target_ids = ['S105', 'S104']
        rain_value = 0
        found = False
        for target_id in target_ids:
            for i, station in enumerate(stations):
                if station['id'] == target_id:
                    rain_value = readings[i]['value']
                    found = True
                    break
            if found: break
        if rain_value > 0: return "🌧️ Rain", "Roads might be wet."
        return "☀️ Clear", "No rain detected."
    except: return "⚠️ Unavailable", "Could not load weather."

weather_status, weather_desc = get_weather()
st.info(f"**Weather at Causeway:** {weather_status}\n\n_{weather_desc}_")

# 3. LIVE IMAGE WITH CALIBRATION LINE
st.write("---")
if os.path.exists("latest_traffic.jpg"):
    img = cv2.imread("latest_traffic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    # Draw the green line that the bot uses to separate Johor (Left) and Woodlands (Right)
    start, end = (int(w * tx), int(h * ty)), (int(w * bx), int(h * by))
    cv2.line(img, start, end, (0, 255, 0), 10)
    st.image(img, caption="Live View: Johor (Left) | Woodlands (Right)", use_container_width=True)

# 4. DATA DISPLAY & ANALYTICS
try:
    df = pd.read_csv("data.csv")
    if not df.empty:
        df['Time'] = pd.to_datetime(df['Time'])
        latest = df.iloc[-1]
        st.write(f"**Last Update:** {latest['Time']}")

        col1, col2 = st.columns(2)
        
        # JOHOR CARD (Left Side of Line)
        with col1:
            val_j = int(latest["To_Johor"])
            st.metric("To Johor 🇲🇾", val_j)
            if val_j < 25: st.success("✅ CLEAR")
            elif val_j < 50: st.warning("⚠️ MODERATE") 
            else: st.error("🛑 JAM")
            
        # WOODLANDS CARD (Right Side of Line)
        with col2:
            val_w = int(latest["To_Woodlands"])
            st.metric("To Woodlands 🇸🇬", val_w)
            if val_w < 25: st.success("✅ CLEAR")
            elif val_w < 50: st.warning("⚠️ MODERATE") 
            else: st.error("🛑 JAM")

        # 📈 24-HOUR TREND (LINE CHART)
        st.write("---")
        st.subheader("📈 24-Hour Traffic Trend")
        chart_data = df.tail(48).copy()
        chart_data["Display_Time"] = chart_data["Time"].dt.strftime("%H:%M")
        st.line_chart(chart_data.set_index("Display_Time")[["To_Johor", "To_Woodlands"]])

        # 📊 SUB-BUSINESS PROBLEM 1: HORIZONTAL BAR CHART
        st.write("---")
        st.subheader("📊 Current Traffic Distribution")
        dist_df = pd.DataFrame({
            'Direction': ['To Johor', 'To Woodlands'],
            'Vehicles': [val_j, val_w]
        })
        st.bar_chart(dist_df.set_index('Direction'), horizontal=True)

except FileNotFoundError:
    st.error("Waiting for data.csv to be generated...")

if st.button("Refresh Page"):
    st.rerun()