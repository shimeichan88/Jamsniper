import os
import requests
import pandas as pd
from datetime import datetime
import pytz
from ultralytics import YOLO
import cv2

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
    # 1. POWER: Extra-Large model + High resolution (1280) to see distant cars
    model = YOLO("yolov8x.pt") 
    results = model("latest_traffic.jpg", conf=0.15, iou=0.75, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    top_x, top_y = w * TX, h * TY
    bottom_x, bottom_y = w * BX, h * BY
    
    j_val, w_val = 0, 0
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_area = float((x2 - x1) * (y2 - y1))
        
        line_x = bottom_x + (top_x - bottom_x) * ((cy - bottom_y) / (top_y - bottom_y))
        
        # --- 2. THE 80% CONFIDENCE MATH ---
        # Perspective factor: 0.0 at bottom, 1.0 at horizon
        norm_h = max(0.01, min(1.0, 1.0 - (cy / h)))
        
        # Weighting: Small area (far away) gets higher weight.
        weight = 1.2 + (850 / (box_area + 25)) * (norm_h ** 1.5)
        
        # 3. THE SAFETY CAP: Prevents massive errors.
        weight = min(weight, 35.0) 
        
        if cx < line_x: j_val += weight
        else: w_val += weight
        
    return int(j_val), int(w_val)

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
                if "Weather" not in df_old.columns:
                    df_old["Weather"] = "N/A"
                df = pd.concat([df_old, new_df], ignore_index=True)
            except:
                df = new_df
        else:
            df = new_df
            
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
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "text": msg, 
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True 
                }
                requests.post(url, json=payload)
