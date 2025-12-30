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

# Load Model
model = YOLO('yolov8m.pt') 

# --- SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state['traffic_data'] = None 
if 'history' not in st.session_state:
    st.session_state['history'] = []

def fetch_and_analyze():
    """Downloads image and runs AI."""
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
        
        if not target_link:
            return None

        img_resp = requests.get(target_link)
        img = Image.open(BytesIO(img_resp.content))
        
        # TUNING UPDATE: 
        # conf=0.10 -> Ignores weak noise (signboards)
        # iou=0.6   -> Separates bikes from cars better
        results = model(img, imgsz=1024, conf=0.10, iou=0.6, classes=[2, 3, 5, 7])
        
        return {"image": img, "results": results[0]}
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def draw_interface(data, shift, tilt):
    """Draws lines and boxes with logic filters."""
    img = data['image'].copy() 
    results = data['results']
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # 1. DRAW DIVIDER LINE
    base_top = width * 0.60
    base_bottom = width * 0.40
    top_x = base_top + (width * shift) + (width * tilt)
    bottom_x = base_bottom + (width * shift) - (width * tilt)
    draw.line([(top_x, 0), (bottom_x, height)], fill="yellow", width=5)
    
    to_johor = 0
    to_sg = 0
    slope = (bottom_x - top_x) / height
    
    # 2. PROCESS BOXES
    for box in results.boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        class_id = int(box.cls[0]) # 2=Car, 3=Bike, 5=Bus, 7=Truck
        
        # --- FILTER 1: THE IGNORE ZONE (Bottom Left) ---
        # If the object is in the bottom 25% AND left 25% of the screen, ignore it.
        # This kills the signboards/railings.
        if center_y > (height * 0.75) and center_x < (width * 0.25):
            continue 

        # --- LOGIC: Left or Right of Line? ---
        divider_x = top_x + (slope * center_y)
        
        # Determine Color based on Class
        if class_id == 3: # Motorcycle
            box_color = "#0099ff" # Blue for Bikes
            width_line = 2
        elif center_x < divider_x:
            box_color = "#00ff00" # Green (To Johor)
            width_line = 2
        else:
            box_color = "#ff0000" # Red (To SG)
            width_line = 2

        # Count Logic
        if center_x < divider_x:
            to_johor += 1
        else:
            to_sg += 1
            
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=width_line)
            
    return img, to_johor, to_sg

# --- WEBSITE ---
st.set_page_config(layout="wide", page_title="JamSniper Pro")
st.title("🚦 JamSniper: Pro Dashboard")

# --- SIDEBAR ---
st.sidebar.header("1. Calibration")
shift_val = st.sidebar.slider("↔️ Position", -0.5, 0.5, 0.05, 0.01)
tilt_val = st.sidebar.slider("🔄 Tilt", -0.5, 0.5, 0.25, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("2. Live Feed")

if st.sidebar.button("📸 Refresh & Analyze", type="primary"):
    with st.spinner("Processing Feed..."):
        data = fetch_and_analyze()
        if data:
            st.session_state['traffic_data'] = data
            
            # Save to History
            _, j_count, s_count = draw_interface(data, shift_val, tilt_val)
            current_time = datetime.now().strftime("%H:%M:%S")
            st.session_state['history'].append({
                "Time": current_time,
                "To Johor": j_count,
                "To Singapore": s_count
            })
            
        else:
            st.error("Camera Offline.")

# --- MAIN DISPLAY ---
if st.session_state['traffic_data']:
    processed_img, count_johor, count_sg = draw_interface(st.session_state['traffic_data'], shift_val, tilt_val)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.image(processed_img, use_column_width=True, caption="Calibrated View")
        
    with col2:
        m1, m2 = st.columns(2)
        m1.metric("To Johor", f"{count_johor}", delta_color="inverse")
        m2.metric("To Singapore", f"{count_sg}", delta_color="inverse")
        
        st.divider()
        st.caption("🔵 Blue Boxes = Motorcycles")
        
        if len(st.session_state['history']) > 0:
            st.subheader("📈 Trend")
            df = pd.DataFrame(st.session_state['history'])
            st.line_chart(df.set_index("Time"))
else:
    st.info("👈 Click 'Refresh & Analyze' to start.")