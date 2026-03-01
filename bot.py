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
    # 1. UPGRADED MODEL: yolov8x is significantly better at small objects/low light
    model = YOLO("yolov8x.pt") 
    
    # 2. STRICT FILTERS: 
    # conf=0.30 kills "ghost" noise on empty roads.
    # iou=0.45 prevents the "1857" error by merging overlapping boxes.
    results = model("latest_traffic.jpg", conf=0.30, iou=0.45, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    top_x, top_y = w * TX, h * TY
    bottom_x, bottom_y = w * BX, h * BY
    
    # 3. TIME-OF-DAY LOGIC (Singapore Time)
    sgt = pytz.timezone('Asia/Singapore')
    current_hour = datetime.now(sgt).hour
    # Increase multiplier between 7 PM and 7 AM to account for invisible dark cars
    is_night = (current_hour >= 19 or current_hour <= 7)
    night_boost = 5.0 if is_night else 0.0
    
    j_val, w_val = 0, 0
    
    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        
        # Determine which side of the divider the car is on
        line_x = bottom_x + (top_x - bottom_x) * ((cy - bottom_y) / (top_y - bottom_y))
        
        # 4. STABLE PERSPECTIVE WEIGHTING
        # norm_h is 0.0 at the bottom (close) and 1.0 at the top (far edge)
        norm_h = max(0, min(1.0, 1.0 - (cy / h)))
        
        # This curve ensures 1 car at the far edge = ~13 in Day, ~18 at Night.
        # It prevents a single car from ever triggering a "200+" count.
        weight = 1.2 + (norm_h ** 2) * (12.0 + night_boost)
        
        if cx < line_x: j_val += weight
        else: w_val += weight
        
    return int(j_val), int(w_val)

def get_status(count):
    # Adjusting thresholds slightly to match the new stable weights
    if count < 40: return "CLEAR"
    elif count < 120: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    requests.post(url, json=payload)

if __name__ == "__main__":
    if download_traffic_image():
        johor_count, woodlands_count = analyze_traffic()
        weather_info = get_weather()
        
        j_status, w_status = get_status(johor_count), get_status(woodlands_count)
        current_status_string = f"{j_status}-{w_status}"
        
        last_status_string = ""
        if os.path.exists("data.csv"):
            try:
                df_old = pd.read_csv("data.csv")
                if not df_old.empty:
                    last_row = df_old.iloc[-1]
                    old_j = get_status(last_row['To_Johor'])
                    old_w = get_status(last_row['To_Woodlands'])
                    last_status_string = f"{old_j}-{old_w}"
            except: pass

        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        new_row = pd.DataFrame([{"Time": now, "To_Johor": johor_count, "To_Woodlands": woodlands_count, "Weather": weather_info}])
        
        if os.path.exists("data.csv"):
            df = pd.concat([pd.read_csv("data.csv"), new_row], ignore_index=True)
        else: 
            df = new_row
        df.to_csv("data.csv", index=False)

        # Alerts Telegram ONLY if Traffic Status (Clear/Mod/Jam) changes
        if current_status_string != last_status_string:
            msg = (f"🚦 <b>Causeway Status Change</b>\n\n"
                   f"🇲🇾 To Johor: {johor_count} ({j_status})\n"
                   f"🇸🇬 To Woodlands: {woodlands_count} ({w_status})\n\n"
                   f"🕒 {now} | {weather_info}\n"
                   f"<a href='https://jamsniper.streamlit.app/'>View Live Dashboard</a>")
            send_telegram(msg)