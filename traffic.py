import streamlit as st
import pandas as pd
import requests
import os

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="JamSniper", layout="centered")
st.title("🚦 JamSniper: Causeway Traffic")

# 2. WEATHER FUNCTION
def get_weather():
    try:
        url = "https://api.data.gov.sg/v1/environment/rainfall"
        data = requests.get(url).json()
        stations = data['metadata']['stations']
        readings = data['items'][0]['readings']
        
        target_ids = ['S105', 'S104'] # Priority: Causeway -> Woodlands Ave 9
        rain_value = 0
        found = False
        
        for target_id in target_ids:
            for i, station in enumerate(stations):
                if station['id'] == target_id:
                    rain_value = readings[i]['value']
                    found = True
                    break
            if found: break
        
        if not found: return "☁️ Unknown", "All nearby sensors offline"
        if rain_value == 0: return "☀️ Clear", "No rain detected."
        elif rain_value < 5: return "🌧️ Light Rain", "Roads might be wet."
        else: return "⛈️ Heavy Rain", "Visibility is poor!"
        
    except Exception:
        return "⚠️ Unavailable", "Could not load weather."

# 3. DISPLAY WEATHER
weather_status, weather_desc = get_weather()
st.info(f"**Weather at Causeway:** {weather_status}\n\n_{weather_desc}_")

# 4. SHOW THE LIVE IMAGE
st.write("---")
if os.path.exists("latest_traffic.jpg"):
    st.image("latest_traffic.jpg", caption="Live View from Robot Eyes", use_column_width=True)
else:
    st.info("Waiting for the first image update... (Run the bot!)")

# 5. LOAD & DISPLAY TRAFFIC DATA
try:
    df = pd.read_csv("data.csv")
    
    if not df.empty:
        df['Time'] = pd.to_datetime(df['Time'])
        latest = df.iloc[-1]
        
        st.write(f"**Last Update:** {latest['Time']}")

        # --- SCORECARDS (Updated Threshold: 50) ---
        col1, col2 = st.columns(2)
        
        # Card 1: To Johor
        with col1:
            st.metric("To Johor", int(latest["To_Johor"]))
            if latest["To_Johor"] < 25: st.success("✅ CLEAR")
            elif latest["To_Johor"] < 50: st.warning("⚠️ MODERATE") # Changed to 50
            else: st.error("🛑 JAM")
            
        # Card 2: To Woodlands
        with col2:
            st.metric("To Woodlands", int(latest["To_Woodlands"]))
            if latest["To_Woodlands"] < 25: st.success("✅ CLEAR")
            elif latest["To_Woodlands"] < 50: st.warning("⚠️ MODERATE") # Changed to 50
            else: st.error("🛑 JAM")
            
        # --- CHART (FIXED X-AXIS) ---
        st.write("---")
        st.subheader("📈 24-Hour Trend")
        
        chart_data = df.tail(48).copy()
        chart_data["Display_Time"] = chart_data["Time"].dt.strftime("%H:%M")
        st.line_chart(chart_data.set_index("Display_Time")[["To_Johor", "To_Woodlands"]])

    else:
        st.warning("Data file is empty. Wait for the bot to run.")

except FileNotFoundError:
    st.error("No data found! Please check if your 'JamSniper Bot' is running in GitHub Actions.")

# 6. REFRESH BUTTON
if st.button("Refresh Data"):
    st.rerun()