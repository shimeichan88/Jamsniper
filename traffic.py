import streamlit as st
import pandas as pd
import requests
import os
import cv2
from datetime import datetime
import pytz

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="JamSniper", layout="centered")

# --- SIDEBAR: YOUR 2 MANUAL SLIDERS ---
st.sidebar.title("📏 Divider Calibration")
shift_top = st.sidebar.slider("Top Point (Horizon)", 0.0, 1.0, 0.70, 0.01)
shift_bottom = st.sidebar.slider("Bottom Point (Near)", 0.0, 1.0, 0.35, 0.01)

st.sidebar.divider()
st.sidebar.subheader("✅ Copy to your Bot logic:")
st.sidebar.code(f"SHIFT_TOP = {shift_top}\nSHIFT_BOTTOM = {shift_bottom}")

# 2. DYNAMIC WEATHER & TEMP (REAL DATA)
def get_live_weather():
    try:
        # Pulling real-time temp and rain code for Singapore
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        res = requests.get(url).json()
        temp = res['current_weather']['temperature']
        code = res['current_weather']['weathercode']
        
        # Mapping codes to real rain status
        rain_status = "No Rain Detected"
        if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            rain_status = "Rain Detected"
            
        return f"{temp}°C", rain_status
    except:
        return "N/A°C", "Weather Unavailable"

# 3. LIVE IMAGE WITH YOUR SLIDER LINE
st.title("🚦 JamSniper: Causeway Traffic")
if os.path.exists("latest_traffic.jpg"):
    img = cv2.imread("latest_traffic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Drawing the line based on your sliders
    top_x, bottom_x = int(w * shift_top), int(w * shift_bottom)
    cv2.line(img, (top_x, 0), (bottom_x, h), (0, 255, 0), 10)
    st.image(img, use_container_width=True)

# 4. TELEGRAM ALERT (DYNAMIC)
try:
    df = pd.read_csv("data.csv")
    if not df.empty:
        latest = df.iloc[-1]
        j_count, w_count = int(latest["To_Johor"]), int(latest["To_Woodlands"])
        
        # Real-time status logic
        j_status = "CLEAR" if j_count < 25 else "MODERATE" if j_count < 50 else "JAM"
        w_status = "CLEAR" if w_count < 25 else "MODERATE" if w_count < 50 else "JAM"
        
        if st.button("Send Telegram Now"):
            real_temp, real_rain = get_live_weather()
            now_str = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%d %H:%M")
            
            # Everything below is pulled from variables
            msg = (f"🚦 <b>Causeway Traffic Update</b> 🚦\n\n"
                   f"🇲🇾 To Johor: {j_count} ({j_status})\n"
                   f"🇸🇬 To Woodlands: {w_count} ({w_status})\n\n"
                   f"🕒 {now_str} | {real_temp} | {real_rain}\n"
                   f"<a href='https://jamsniper.streamlit.app/'>View Live Cameras Here</a>")
            
            url = f"https://api.telegram.org/bot{os.environ.get('TELEGRAM_TOKEN')}/sendMessage"
            requests.post(url, json={"chat_id": os.environ.get("TELEGRAM_CHAT_ID"), "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True})
            st.success(f"Sent: {real_temp} | {real_rain}")

except FileNotFoundError:
    st.error("No data.csv found.")