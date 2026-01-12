import streamlit as st
import pandas as pd
import requests

# 1. CONFIGURATION
st.set_page_config(page_title="JamSniper Dashboard", layout="centered")
st.title("🚦 JamSniper: Causeway Traffic")

# 2. WEATHER FUNCTION
def get_weather():
    try:
        # Fetch real-time rain data from Singapore NEA
        url = "https://api.data.gov.sg/v1/environment/rainfall"
        data = requests.get(url).json()
        
        # Station S105 is "Admiralty Road West" (Closest to Causeway)
        stations = data['metadata']['stations']
        readings = data['items'][0]['readings']
        
        rain_value = 0
        found = False
        
        for i, station in enumerate(stations):
            if station['id'] == 'S105':
                rain_value = readings[i]['value']
                found = True
                break
        
        # Logic: 0 = Clear, <5 = Light Rain, >5 = Heavy Rain
        if not found: return "☁️ Unknown", "Sensor offline"
        if rain_value == 0: return "☀️ Clear", "No rain detected."
        elif rain_value < 5: return "🌧️ Light Rain", "Roads might be wet."
        else: return "⛈️ Heavy Rain", "Visibility is poor!"
        
    except Exception:
        return "⚠️ Error", "Could not load weather."

# 3. DISPLAY WEATHER
weather_status, weather_desc = get_weather()
st.info(f"**Weather at Causeway:** {weather_status}\n\n_{weather_desc}_")

# 4. LOAD DATA
try:
    df = pd.read_csv("data.csv")
    
    # Check if we have data
    if not df.empty:
        latest = df.iloc[-1]
        timestamp = latest["Time"]
        
        # Handle older CSVs that might not have split columns yet
        johor = latest.get("To_Johor", 0)
        woodlands = latest.get("To_Woodlands", 0)
        
        st.write(f"**Last Update:** {timestamp}")

        # 5. METRICS (SPLIT)
        col1, col2 = st.columns(2)
        
        # -- TO JOHOR --
        with col1:
            st.metric("To Johor", int(johor))
            if johor < 25: st.success("✅ CLEAR")
            elif johor < 45: st.warning("⚠️ MODERATE")
            else: st.error("🛑 JAM")
            
        # -- TO WOODLANDS --
        with col2:
            st.metric("To Woodlands", int(woodlands))
            if woodlands < 25: st.success("✅ CLEAR")
            elif woodlands < 45: st.warning("⚠️ MODERATE")
            else: st.error("🛑 JAM")
            
        # 6. CHART (Reverted to Line Chart)
        st.write("---")
        st.subheader("📈 24-Hour Trend")
        
        # Ensure we interpret 'Time' correctly for the chart
        # We assume the last 48 entries = ~24 hours
        chart_data = df.tail(48).copy()
        st.line_chart(chart_data[["To_Johor", "To_Woodlands"]])

    else:
        st.warning("Data file is empty. Wait for the bot to run.")

except FileNotFoundError:
    st.error("No data.csv found yet. Please run the bot in GitHub Actions.")

# 7. REFRESH BUTTON
if st.button("Refresh Data"):
    st.rerun()