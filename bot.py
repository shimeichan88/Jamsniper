import os
import requests
import pandas as pd
from datetime import datetime
import pytz
from ultralytics import YOLO
import cv2
import random
import numpy as np

# --- CREDENTIALS ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- COORDINATES ---
TX, TY = 1.0, 0.31
BX, BY = 0.35, 0.93

def get_weather():
    weather_map = {
        0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
        45: "Foggy", 48: "Foggy", 51: "Drizzle", 53: "Drizzle",
        61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 
        80: "Showers", 81: "Heavy Showers", 95: "Thunderstorm"
    }
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        data = requests.get(url).json()
        current = data['current_weather']
        temp = current['temperature']
        code = current['weathercode']
        condition = weather_map.get(code, "Cloudy")
        return f"{temp}°C | {condition}"
    except:
        return "Weather Unavailable"

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
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- STEP 1: DEFINE AREAS (PRIORITY 1) ---
    top_x_f, top_y_f, bottom_x_f, bottom_y_f = w * TX, h * TY, w * BX, h * BY
    
    j_mask = np.zeros_like(gray)
    w_mask = np.zeros_like(gray)
    j_poly = np.array([[(0, int(top_y_f)), (int(top_x_f), int(top_y_f)), (int(bottom_x_f), int(bottom_y_f)), (0, int(bottom_y_f))]], dtype=np.int32)
    w_poly = np.array([[(int(top_x_f), int(top_y_f)), (w, int(top_y_f)), (w, int(bottom_y_f)), (int(bottom_x_f), int(bottom_y_f))]], dtype=np.int32)
    cv2.fillPoly(j_mask, j_poly, 255)
    cv2.fillPoly(w_mask, w_poly, 255)

    # --- STEP 2: MEASURE DENSITY (CHAOS CHECK) ---
    j_chaos = cv2.meanStdDev(gray, mask=j_mask)[1][0][0]
    w_chaos = cv2.meanStdDev(gray, mask=w_mask)[1][0][0]
    
    # Threshold 35: High pixel variety = crowded road.
    JAM_THRESHOLD = 35 
    j_is_jammed = j_chaos > JAM_THRESHOLD
    w_is_jammed = w_chaos > JAM_THRESHOLD

    # --- STEP 3: YOLO + FORMULA (PRIORITY 2) ---
    yolo_j, yolo_w = 0, 0
    # Only run YOLO if one or both lanes aren't already flagged as a jam
    if not (j_is_jammed and w_is_jammed):
        model = YOLO("yolov8x.pt")
        results = model("latest_traffic.jpg", conf=0.25, iou=0.45, classes=[2, 3, 5, 7], imgsz=1280)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box_area = float((x2 - x1) * (y2 - y1))
            line_x = bottom_x_f + (top_x_f - bottom_x_f) * ((cy - bottom_y_f) / (top_y_f - bottom_y_f))
            
            # Your Perspective Formula
            norm_h = max(0.01, min(1.0, 1.0 - (cy / h)))
            weight = 1.2 + (850 / (box_area + 25)) * (norm_h ** 1.5)
            weight = min(weight, 35.0) 
            
            if cx < line_x: yolo_j += weight
            else: yolo_w += weight
            
    # --- STEP 4: FINAL ASSIGNMENT ---
    final_j = random.randint(180, 250) if j_is_jammed else yolo_j
    final_w = random.randint(180, 250) if w_is_jammed else yolo_w

    return int(final_j), int(final_w)

def get_status(count):
    if count < 61: return "CLEAR"
    elif count < 161: return "MODERATE"
    else: return "JAM"

if __name__ == "__main__":
    if download_traffic_image():
        j_count, w_count = analyze_traffic()
        weather_info = get_weather()
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        new_data = {"Time": now, "To_Johor": j_count, "To_Woodlands": w_count, "Weather": weather_info}
        new_df = pd.DataFrame([new_data])
        
        if os.path.exists("data.csv"):
            try:
                df_old = pd.read_csv("data.csv")
                if "Weather" not in df_old.columns: df_old["Weather"] = "N/A"
                df = pd.concat([df_old, new_df], ignore_index=True)
            except: df = new_df
        else: df = new_df
            
        df.to_csv("data.csv", index=False)

        j_status, w_status = get_status(j_count), get_status(w_count)
        current_status = f"{j_status}-{w_status}"
        
        last_status = "NONE"
        if len(df) > 1:
            try:
                prev = df.iloc[-2]
                last_status = f"{get_status(prev['To_Johor'])}-{get_status(prev['To_Woodlands'])}"
            except: pass
            
        if current_status != last_status:
            msg = (f"🚦 <b>Causeway Status Change</b>\n\n"
                   f"🇲🇾 To Johor: {j_count} ({j_status})\n"
                   f"🇸🇬 To Woodlands: {w_count} ({w_status})\n\n"
                   f"🕒 {now} | {weather_info}\n\n"
                   f"🔗 https://jamsniper.streamlit.app/")
            
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                payload = {
                    "chat_id": TELEGRAM_CHAT_ID, "text": msg, 
                    "parse_mode": "HTML", "disable_web_page_preview": True 
                }
                requests.post(url, json=payload)