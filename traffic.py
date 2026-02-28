import streamlit as st
import pandas as pd
import requests
import os
import cv2
from datetime import datetime
import pytz

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="JamSniper", layout="centered")
st.title("🚦 JamSniper: Causeway Traffic")

# --- SIDEBAR: 4 SLIDERS (Restored with your points) ---
st.sidebar.title("📏 Line Calibration")
tx = st.sidebar.slider("Top X", 0.0, 1.0, 1.0)
ty = st.sidebar.slider("Top Y", 0.0, 1.0, 0.31)
bx = st.sidebar.slider("Bottom X", 0.0, 1.0, 0.35)
by = st.sidebar.slider("Bottom Y", 0.0, 1.0, 0.93)

# 2. WEATHER FUNCTION (Restored your original logic)
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
        if not found: return "☁️ Unknown", "All nearby sensors offline"
        if rain_value == 0: return "☀️ Clear", "No rain detected."
        elif rain_value < 5: return "🌧️ Light Rain", "Roads might be wet."
        else: return "⛈️ Heavy Rain", "Visibility is poor!"
    except: return "⚠️ Unavailable", "Could not load weather."

weather_status, weather_desc = get_weather()
st.info(f"**Weather at Causeway:** {weather_status}\n\n_{weather_desc}_")

# 3. LIVE IMAGE (Restored with your divider)
st.write("---")
if os.path.exists("latest_traffic.jpg"):
    img = cv2.imread("latest_traffic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    start, end = (int(w * tx), int(h * ty)), (int(w * bx), int(h * by))
    cv2.line(img, start, end, (0, 255, 0), 10)
    st.image(img, caption="Live View from Robot Eyes", use_container_width=True)

# 4. LOAD & DISPLAY TRAFFIC DATA (Restored Analytics)
try:
    df = pd.read_csv("data.csv")
    if not df.empty:
        df['Time'] = pd.to_datetime(df['Time'])
        latest = df.iloc[-1]
        st.write(f"**Last Update:** {latest['Time']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("To Johor", int(latest["To_Johor"]))
            j_stat = "✅ CLEAR" if latest["To_Johor"] < 25 else "⚠️ MODERATE" if latest["To_Johor"] < 50 else "🛑 JAM"
            st.write(j_stat)
        with col2:
            st.metric("To Woodlands", int(latest["To_Woodlands"]))
            w_stat = "✅ CLEAR" if latest["To_Woodlands"] < 25 else "⚠️ MODERATE" if latest["To_Woodlands"] < 50 else "🛑 JAM"
            st.write(w_stat)

        if st.button("Send Telegram Now"):
            now = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%d %H:%M")
            msg = (f"🚦 <b>Causeway Traffic Update</b> 🚦\n\n"
                   f"🇲🇾 To Johor: {int(latest['To_Johor'])} ({j_stat.split()[-1]})\n"
                   f"🇸🇬 To Woodlands: {int(latest['To_Woodlands'])} ({w_stat.split()[-1]})\n\n"
                   f"🕒 {now} | 27.7°C | {weather_status}\n"
                   f"<a href='https://jamsniper.streamlit.app/'>View Live Cameras Here</a>")
            requests.post(f"https://api.telegram.org/bot{os.environ.get('TELEGRAM_TOKEN')}/sendMessage", 
                          json={"chat_id": os.environ.get("TELEGRAM_CHAT_ID"), "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True})
            st.success("Sent!")

        st.write("---")
        st.subheader("📈 24-Hour Traffic Trend")
        chart_data = df.tail(48).copy()
        chart_data["Display_Time"] = chart_data["Time"].dt.strftime("%H:%M")
        st.line_chart(chart_data.set_index("Display_Time")[["To_Johor", "To_Woodlands"]])

except FileNotFoundError: st.error("No data found!")
if st.button("Refresh Data"): st.rerun()