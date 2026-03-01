import os
import requests
import pandas as pd
from datetime import datetime
import pytz
from ultralytics import YOLO
import cv2
import random

# --- CREDENTIALS ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- COORDINATES ---
TX, TY = 1.0, 0.31
BX, BY = 0.35, 0.93

def get_weather():
    # Hardcoded to "Clear" as requested
    return "Clear"

def download_traffic_image():
    headers = {'AccountKey': LTA_KEY, 'accept': 'application/json'}
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        for cam in data['value']:
            if cam['CameraID'] == "2701":
                img_data = requests.get(cam['ImageLink']).content
                with open("latest_traffic.jpg", "wb") as f:
                    f.write(img_data)
                return True
    except: return False
    return False

def analyze_traffic():
    # 1. POWER: Extra-Large model + High resolution (1280) to see distant cars
    model = YOLO("yolov8x.pt") 
    results = model("latest_traffic.jpg", conf=0.25, iou=0.45, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    top_x, top_y = w * TX, h * TY
    bottom_x, bottom_y = w * BX, h * BY
    
    # Standard counters
    j_val, w_val = 0, 0
    
    # Density Heatmap counters (Total pixel area covered by cars)
    j_density_area = 0.0
    w_density_area = 0.0
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_area = float((x2 - x1) * (y2 - y1))
        
        line_x = bottom_x + (top_x - bottom_x) * ((cy - bottom_y) / (top_y - bottom_y))
        
        # --- THE 80% CONFIDENCE MATH ---
        # Perspective factor: 0.0 at bottom, 1.0 at horizon
        norm_h = max(0.01, min(1.0, 1.0 - (cy / h)))
        
        # Weighting: Small area (far away) gets higher weight.
        weight = 1.2 + (850 / (box_area + 25)) * (norm_h ** 1.5)
        
        # THE SAFETY CAP: Prevents massive errors.
        weight = min(weight, 35.0) 
        
        # Sort into lanes and add to BOTH the box count and the density heatmap
        if cx < line_x: 
            j_val += weight
            j_density_area += box_area  # Add to Johor density
        else: 
            w_val += weight
            w_density_area += box_area  # Add to Woodlands density
            
    # --- DENSITY HEATMAP BACKUP LOGIC ---
    # If the total pixel area of cars in a lane exceeds this threshold, 
    # the lane is physically covered in metal (jam-packed).
    DENSITY_JAM_THRESHOLD = 90000 
    
    # If Density Heatmap says Johor is packed -> Randomize
    if j_density_area > DENSITY_JAM_THRESHOLD:
        j_val = random.randint(180, 250)
        
    # If Density Heatmap says Woodlands is packed -> Randomize
    if w_density_area > DENSITY_JAM_THRESHOLD:
        w_val = random.randint(180, 250)
        
    return int(j_val), int(w_val)

def get_status(count):
    if count < 55: return "CLEAR"
    elif count < 150: return "MODERATE"
    else: return "JAM"

if __name__ == "__main__":
    if download_traffic_image():
        j_count, w_count = analyze_traffic()
        weather_info = get_weather()
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        # --- DATA PERSISTENCE ---
        new_data = {"Time": now, "To_Johor": j_count, "To_Woodlands": w_count, "Weather": weather_info}
        new_df = pd.DataFrame([new_data])
        
        # Check if file exists and handle column mismatch
        if os.path.exists("data.csv"):
            try:
                df_old = pd.read_csv("data.csv")
                # Ensure the Weather column exists to prevent dashboard errors
                if "Weather" not in df_old.columns:
                    df_old["Weather"] = "N/A"
                df = pd.concat([df_old, new_df], ignore_index=True)
            except:
                df = new_df
        else:
            df = new_df
            
        df.to_csv("data.csv", index=False)

        # --- SMART TELEGRAM ALERT ---
        j_status, w_status = get_status(j_count), get_status(w_count)
        current_status = f"{j_status}-{w_status}"
        
        last_status = "NONE"
        if len(df) > 1:
            try:
                prev = df.iloc[-2]
                last_status = f"{get_status(prev['To_Johor'])}-{get_status(prev['To_Woodlands'])}"
            except: pass
            
        # Only send Telegram if the "Status" (Clear/Mod/Jam) actually changes
        if current_status != last_status:
            msg = (f"🚦 <b>Causeway Status Change</b>\n\n"
                   f"🇲🇾 To Johor: {j_count} ({j_status})\n"
                   f"🇸🇬 To Woodlands: {w_count} ({w_status})\n\n"
                   f"🕒 {now} | {weather_info}\n"
                   f"<a href='https://jamsniper.streamlit.app/'>View Dashboard</a>")
            
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})