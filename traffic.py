import streamlit as st
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
if "LTA_API_KEY" in st.session_state:
    API_KEY = st.session_state["LTA_API_KEY"]
elif "LTA_API_KEY" in st.secrets:
    API_KEY = st.secrets["LTA_API_KEY"]
else:
    st.error("API Key missing! Please add it in Secrets.")
    st.stop()

model = YOLO('yolov8m.pt') 

# --- SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state['traffic_data'] = None 
if 'history' not in st.session_state:
    st.session_state['history'] = []

def fetch_and_analyze():
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": API_KEY, "accept": "application/json"}
    try:
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
        
        results = model(img, imgsz=1024, conf=0.15, iou=0.6, classes=[2, 3, 5, 7])
        return {"image": img, "results": results[0]}
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def draw_interface(data, shift, tilt):
    img = data['image'].copy() 
    results = data['results']
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 1. DRAW DIVIDER
    base_top = width * 0.60
    base_bottom = width * 0.40
    top_x = base_top + (width * shift) + (width * tilt)
    bottom_x = base_bottom + (width * shift) - (width * tilt)
    draw.line([(top_x, 0), (bottom_x, height)], fill="yellow", width=5)
    
    to_johor = 0
    to_woodlands = 0
    slope = (bottom_x - top_x) / height
    
    # 2. PROCESS BOXES
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_w = x2 - x1
        box_h = y2 - y1
        
        # FILTER: BILLBOARD KILLER
        if center_y > (height * 0.60) and center_x < (width * 0.30):
            continue 
        if (box_w / box_h) > 3.0: 
            continue

        divider_x = top_x + (slope * center_y)
        
        if center_x < divider_x:
            to_johor += 1
            color = "#00ff00" # Green
        else:
            to_woodlands += 1
            color = "#ff0000" # Red
            
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
    return img, to_johor, to_woodlands

# --- WEBSITE LAYOUT ---
st.set_page_config(layout="wide", page_title="JamSniper Phase 2")
st.title("🚦 JamSniper: Live Dashboard")

# SIDEBAR
st.sidebar.header("Calibration")
shift_val = st.sidebar.slider("↔️ Position", -0.5, 0.5, 0.05, 0.01)
tilt_val = st.sidebar.slider("🔄 Tilt", -0.5, 0.5, 0.25, 0.01)
st.sidebar.divider()
if st.sidebar.button("📸 Refresh Feed", type="primary"):
    with st.spinner("Analyzing..."):
        data = fetch_and_analyze()
        if data:
            st.session_state['traffic_data'] = data
            # Record History
            _, j_count, w_count = draw_interface(data, shift_val, tilt_val)
            current_time = datetime.now().strftime("%H:%M")
            st.session_state['history'].append({"Time": current_time, "To Johor": j_count, "To Woodlands": w_count})
        else:
            st.error("Camera Offline")

# MAIN DISPLAY
if st.session_state['traffic_data']:
    processed_img, count_johor, count_woodlands = draw_interface(st.session_state['traffic_data'], shift_val, tilt_val)
    
    col1, col2 = st.columns([0.75, 0.25])
    
    with col1:
        st.image(processed_img, use_column_width=True, caption="Live Analysis")
        
        if len(st.session_state['history']) > 0:
            st.markdown("### 📈 Session Trend")
            df = pd.DataFrame(st.session_state['history'])
            st.line_chart(df.set_index("Time"))

    with col2:
        st.markdown("### 📊 Status")
        
        # CARD 1: TO JOHOR
        st.markdown("---")
        st.write("**To Johor**")
        st.metric("Estimated Number of Vehicles", f"{count_johor}")
        if count_johor < 25:
            st.success("✅ CLEAR")
        elif count_johor < 45:
            st.warning("⚠️ MODERATE")
        else:
            st.error("🛑 JAM")
            
        # CARD 2: TO WOODLANDS
        st.markdown("---")
        st.write("**To Woodlands**")
        st.metric("Estimated Number of Vehicles", f"{count_woodlands}")
        if count_woodlands < 25:
            st.success("✅ CLEAR")
        elif count_woodlands < 45:
            st.warning("⚠️ MODERATE")
        else:
            st.error("🛑 JAM")

else:
    st.info("👈 Click 'Refresh Feed' to start.")