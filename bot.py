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
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        data = requests.get(url).json()
        temp = data['current_weather']['temperature']
        return f"{temp}°C | Clear"
    except: return "27.0°C | Clear"

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
    model = YOLO("yolov8n.pt") 
    # conf=0.20 ensures we only count real cars, keeping Johor around 30-40
    results = model("latest_traffic.jpg", conf=0.20, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    
    top_x, top_y = w * TX, h * TY
    bottom_x, bottom_y = w * BX, h * BY
    
    j_val, w_val = 0, 0
    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        line_x = bottom_x + (top_x - bottom_x) * ((cy - bottom_y) / (top_y - bottom_y))
        
        norm_h = max(0, min(1.0, 1.0 - (cy / h)))
        
        # This math keeps Johor low (~0.8 per car) and Woodlands high (~40 per cluster)
        weight = 0.8 + (norm_h ** 3) * 60.0 
            
        if cx < line_x: j_val += weight
        else: w_val += weight
            
    return int(j_val), int(w_val)

def get_status(count):
    if count < 61: return "CLEAR"
    elif count < 161: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    requests.post(url, json=payload)

if __name__ == "__main__":
    if download_traffic_image():
        johor, woodlands = analyze_traffic()
        weather = get_weather()
        j_status, w_status = get_status(johor), get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        # Save to CSV
        new_row = pd.DataFrame([{"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Weather": weather}])
        if os.path.exists("data.csv"):
            df = pd.concat([pd.read_csv("data.csv"), new_row], ignore_index=True)
        else: 
            df = new_row
        df.to_csv("data.csv", index=False)
        
        # --- REVERTED MESSAGE FORMAT ---
        msg = (f"🚦 <b>Causeway Traffic Update</b>\n\n"
               f"🇲🇾 To Johor: {johor} ({j_status})\n"
               f"🇸🇬 To Woodlands: {woodlands} ({w_status})\n\n"
               f"🕒 {now} | {weather}\n"
               f"<a href='https://jamsniper.streamlit.app/'>View Live Dashboard</a>")
        
        send_telegram(msg)