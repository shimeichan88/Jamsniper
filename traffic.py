import streamlit as st
import pandas as pd
import requests
import os
import cv2
from datetime import datetime
import pytz

# 1. PAGE SETUP
st.set_page_config(page_title="JamSniper", layout="centered")

# --- SIDEBAR: 4 SLIDERS (Updated to your coordinates) ---
st.sidebar.title("📏 Line Calibration")
st.sidebar.write("Fine-tune your divider position:")

# Top Point: Updated to TX=1.0, TY=0.31
st.sidebar.subheader("Top Point")
tx = st.sidebar.slider("Top X", 0.0, 1.0, 1.0)
ty = st.sidebar.slider("Top Y", 0.0, 1.0, 0.31)

# Bottom Point: Updated to BX=0.35, BY=0.93
st.sidebar.subheader("Bottom Point")
bx = st.sidebar.slider("Bottom X", 0.0, 1.0, 0.35)
by = st.sidebar.slider("Bottom Y", 0.0, 1.0, 0.93)

# (Calibration code display line DELETED as requested)

# 2. DYNAMIC WEATHER (Real Data)
def get_live_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        res = requests.get(url).json()
        temp = res['current_weather']['temperature']
        code = res['current_weather']['weathercode']
        rain = "Rain Detected" if code >= 51 else "No Rain Detected"
        return f"{temp}°C", rain
    except: return "27.0°C", "Weather Unavailable"

# 3. LIVE IMAGE WITH YOUR COORDINATES
st.title("🚦 JamSniper: Causeway Traffic")
if os.path.exists("latest_traffic.jpg"):
    img = cv2.imread("latest_traffic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Using your 4 points for the green line
    start = (int(w * tx), int(h * ty))
    end = (int(w * bx), int(h * by))
    
    cv2.line(img, start, end, (0, 255, 0), 10)
    st.image(img, caption="Live View with Manual Divider", use_container_width=True)

# 4. DATA & TELEGRAM
try:
    df = pd.read_csv("data.csv")
    if not df.empty:
        latest = df.iloc[-1]
        j_val, w_val = int(latest["To_Johor"]), int(latest["To_Woodlands"])
        
        c1, c2 = st.columns(2)
        c1.metric("To Johor", j_val)
        c2.metric("To Woodlands", w_val)
        
        if st.button("Send Telegram Now"):
            t, r = get_live_weather()
            now = datetime.now(pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%d %H:%M")
            j_s = "JAM" if j_val > 50 else "MODERATE" if j_val > 25 else "CLEAR"
            w_s = "JAM" if w_val > 50 else "MODERATE" if w_val > 25 else "CLEAR"

            msg = (f"🚦 <b>Causeway Traffic Update</b> 🚦\n\n"
                   f"🇲🇾 To Johor: {j_val} ({j_s})\n"
                   f"🇸🇬 To Woodlands: {w_val} ({w_s})\n\n"
                   f"🕒 {now} | {t} | {r}\n"
                   f"<a href='https://jamsniper.streamlit.app/'>View Live Cameras Here</a>")
            
            requests.post(f"https://api.telegram.org/bot{os.environ.get('TELEGRAM_TOKEN')}/sendMessage", 
                          json={"chat_id": os.environ.get("TELEGRAM_CHAT_ID"), "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True})
            st.success("Sent!")
except: st.error("Waiting for data.csv...")