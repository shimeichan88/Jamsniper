import streamlit as st
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from ultralytics import YOLO
import pandas as pd
from datetime import datetime, timedelta

# --- CONFIGURATION ---
if "LTA_API_KEY" in st.session_state:
    API_KEY = st.session_state["LTA_API_KEY"]
elif "LTA_API_KEY" in st.secrets:
    API_KEY = st.secrets["LTA_API_KEY"]
else:
    st.error("API Key missing! Please add it in Secrets.")
    st.stop()

CSV_URL = "https://github.com/shimeichan88/Jamsniper/raw/refs/heads/main/data.csv"
WEATHER_URL = "https://api.data.gov.sg/v1/environment/rainfall"

# Load Model
model = YOLO('yolov8m.pt') 

# --- SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state['traffic_data'] = None 

# --- DATA LOADER ---
@st.cache_data(ttl=300)
def load_history():
    try:
        df = pd.read_csv(CSV_URL)
        df['Time'] = pd.to_datetime(df['Time'])
        cutoff_time = datetime.now() - timedelta(hours=24)
        df_recent = df[df['Time'] > cutoff_time].copy()
        df_recent['Time'] = df_recent['Time'].dt.strftime('%H:%M')
        return df_recent.set_index('Time')
    except Exception:
        return pd.DataFrame()

# --- WEATHER CHECKER (NEW) ---
def get_weather():
    try:
        # 1. Ask NEA for data
        resp = requests.get(WEATHER_URL).json()
        
        # 2. Look for Station S105 (Admiralty Road West)
        # Note: We loop because the order of stations changes randomly
        rain_value = 0
        for reading in resp['items'][0]['readings']:
            if reading['station_id'] == 'S105':
                rain_value = reading['value']
                break
        
        # 3. Translate Number to Text
        if rain_value == 0:
            return "☀️ Clear", "Normal"
        elif rain_value < 5:
            return "🌧️ Light Rain", "Caution"
        else:
            return "⛈️ Heavy Rain", "Danger"
            
    except Exception:
        return "☁️ Unknown", "Normal"

# --- AI ANALYZER (HD MODE) ---
def fetch_and_analyze():
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": API_KEY, "accept": "application/json"}
    try:
        # Get Traffic Image
        response = requests.get(url, headers=headers)
        target_link = None
        if response.status_code == 200:
            for img in response.json()['value']:
                if str(img['CameraID']) == "2701":
                    target_link = img['ImageLink']
                    break
        if not target_link: return None
        
        img_resp = requests.get(target_link)
        img = Image.open(BytesIO(img_resp.content))
        
        # Get Weather Data
        weather_text, weather_status = get_weather()
        
        results = model(img, imgsz=1280, conf=0.05, iou=0.7, classes=[2, 3, 5, 7])
        return {
            "image": img, 
            "results": results[0], 
            "weather": weather_text,
            "weather_status": weather_status
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- VISUALIZER ---
def draw_interface(data, shift, tilt):
    img = data['image'].copy() 
    results = data['results']
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    base_top = width * 0.60
    base_bottom = width * 0.40
    top_x = base_top + (width * shift) + (width * tilt)
    bottom_x = base_bottom + (width * shift) - (width * tilt)
    draw.line([(top_x, 0), (bottom_x, height)], fill="yellow", width=5)
    
    to_johor = 0
    to_woodlands = 0
    slope = (bottom_x - top_x) / height
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if center_y > (height * 0.60) and center_x < (width * 0.30): continue 

        divider_x = top_x + (slope * center_y)
        if center_x < divider_x:
            to_johor += 1
            color = "#00ff00"
        else:
            to_woodlands += 1
            color = "#ff0000"
            
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
    return img, to_johor, to_woodlands

# --- LAYOUT ---
st.set_page_config(layout="wide", page_title="JamSniper Pro")

# NEW: Header with Weather
if st.session_state['traffic_data']:
    weather = st.session_state['traffic_data']['weather']
    st.title(f"🚦 JamSniper: {weather}")
else:
    st.title("🚦 JamSniper: Live Dashboard")

st.sidebar.header("Calibration")
shift_val = st.sidebar.slider("↔️ Position", -0.5, 0.5, 0.28, 0.01)
tilt_val = st.sidebar.slider("🔄 Tilt", -0.5, 0.5, 0.43, 0.01)
st.sidebar.divider()

if st.sidebar.button("📸 Refresh Feed", type="primary"):
    with st.spinner("Analyzing Traffic & Weather..."):
        data = fetch_and_analyze()
        if data:
            st.session_state['traffic_data'] = data
        else:
            st.error("Camera Offline")

if st.session_state['traffic_data']:
    processed_img, count_johor, count_woodlands = draw_interface(st.session_state['traffic_data'], shift_val, tilt_val)
    
    col1, col2 = st.columns([0.75, 0.25])
    
    with col1:
        st.image(processed_img, use_column_width=True, caption=f"Live Analysis • {st.session_state['traffic_data']['weather']}")
        
        st.markdown("### 📈 24-Hour Trend")
        history_df = load_history()
        if not history_df.empty:
            # Classic Line Chart
            st.line_chart(history_df[['To_Johor', 'To_Woodlands']])
        else:
            st.info("Waiting for Robot data...")

    with col2:
        st.markdown("### 📊 Status")
        
        # WEATHER CARD
        st.info(f"**Weather at Causeway:**\n\n{st.session_state['traffic_data']['weather']}")
        
        st.markdown("---")
        st.write("**To Johor**")
        st.metric("Score", f"{count_johor}")
        if count_johor < 25: st.success("✅ CLEAR")
        elif count_johor <= 45: st.warning("⚠️ MODERATE")
        else: st.error("🛑 JAM")
            
        st.markdown("---")
        st.write("**To Woodlands**")
        st.metric("Score", f"{count_woodlands}")
        if count_woodlands < 25: st.success("✅ CLEAR")
        elif count_woodlands <= 45: st.warning("⚠️ MODERATE")
        else: st.error("🛑 JAM")

else:
    st.info("👈 Click 'Refresh Feed' to start.")