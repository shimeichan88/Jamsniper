import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from ultralytics import YOLO

# --- CONFIGURATION ---
#API_KEY = "" # <--- PASTE KEY

# This tells the app: "Go look in the Server's Safe for the key"
if "LTA_API_KEY" in st.secrets:
    API_KEY = st.secrets["LTA_API_KEY"]
else:
    st.error("API Key not found!")

model = YOLO('yolov8m.pt') 

# --- SESSION STATE ---
if 'traffic_data' not in st.session_state:
    st.session_state['traffic_data'] = None 

def fetch_and_analyze():
    """Downloads image and runs HYPER-SENSITIVE AI."""
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
        
        # --- THE FIX FOR MISSING CARS ---
        # conf=0.01: Detects almost EVERYTHING (even dark/blurry cars)
        # iou=0.8:   Allows boxes to overlap heavily (Critical for jams)
        # imgsz=1024: Keeps high resolution
        results = model(img, imgsz=1024, conf=0.01, iou=0.8, classes=[2, 3, 5, 7])
        
        return {"image": img, "results": results[0]}
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def draw_interface(data, shift, tilt):
    """Draws lines and boxes INSTANTLY."""
    img = data['image'].copy() 
    results = data['results']
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # --- CALIBRATION MATH ---
    base_top = width * 0.60
    base_bottom = width * 0.40
    
    top_x = base_top + (width * shift) + (width * tilt)
    bottom_x = base_bottom + (width * shift) - (width * tilt)
    
    # Draw Yellow Divider
    draw.line([(top_x, 0), (bottom_x, height)], fill="yellow", width=5)
    
    to_johor = 0
    to_sg = 0
    
    slope = (bottom_x - top_x) / height
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        divider_x = top_x + (slope * center_y)
        
        if center_x < divider_x:
            to_johor += 1
            color = "#00ff00" # Green
        else:
            to_sg += 1
            color = "#ff0000" # Red
            
        # Draw Box (thinner line so we can see more)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
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
if st.sidebar.button("📸 Refresh Traffic Feed", type="primary"):
    with st.spinner("Scanning High-Density Traffic..."):
        data = fetch_and_analyze()
        if data:
            st.session_state['traffic_data'] = data
        else:
            st.error("Camera Offline.")

# --- DISPLAY ---
if st.session_state['traffic_data']:
    processed_img, count_johor, count_sg = draw_interface(st.session_state['traffic_data'], shift_val, tilt_val)
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.image(processed_img, use_column_width=True, caption="Hyper-Sensitive Analysis")
        
    with col2:
        st.markdown("### 📊 Real-Time Status")
        
        # CARD 1: TO JOHOR
        with st.container(border=True):
            st.markdown("**🛫 To Johor (Outbound)**")
            st.metric("Volume", f"{count_johor}")
            if count_johor < 20:
                st.success("✅ CLEAR")
            else:
                st.warning(f"🚗 {count_johor} Cars")

        # CARD 2: TO SINGAPORE
        with st.container(border=True):
            st.markdown("**🛬 To Singapore (Inbound)**")
            st.metric("Volume", f"{count_sg}")
            if count_sg < 20:
                st.success("✅ CLEAR")
            else:
                st.error("🛑 JAMMED")
                
else:
    st.info("👈 Click 'Refresh Traffic Feed' to start.")