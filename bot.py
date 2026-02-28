import os
import requests
import pandas as pd
from datetime import datetime
import pytz
from ultralytics import YOLO
import cv2

# --- SECURE CREDENTIALS ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- MANUAL COORDINATES (Matches your traffic.py calibration) ---
TX = 1.0   # Top X
TY = 0.31  # Top Y
BX = 0.35  # Bottom X
BY = 0.93  # Bottom Y

CONFIDENCE = 0.10   # Sensitivity

def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        data = requests.get(url).json()
        temp = data['current_weather']['temperature']
        code = data['current_weather']['weathercode']
        
        rain_text = "No Rain Detected"
        if code in [51, 53, 55]: rain_text = "Drizzle"
        elif code in [61, 63, 65, 80, 81]: rain_text = "Rain"
        elif code in [66, 67, 82]: rain_text = "Heavy Rain"
            
        return f"{temp}°C | {rain_text}"
    except:
        return "N/A"

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
    results = model("latest_traffic.jpg", conf=CONFIDENCE, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    
    # Calculate pixel positions for your line
    top_x, top_y = w * TX, h * TY
    bottom_x, bottom_y = w * BX, h * BY
    
    j_raw, w_raw = 0, 0
    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        
        # FIXED: Removed the 'if top_y <= cy <= bottom_y' restriction.
        # This math calculates the X-coordinate of the divider line at the specific height (cy) of the car.
        # Formula: x = x2 + (x1 - x2) * (y - y2) / (y1 - y2)
        line_x = bottom_x + (top_x - bottom_x) * ((cy - bottom_y) / (top_y - bottom_y))
        
        # Classification based on which side of the line the center-point (cx) is on
        if cx > line_x: j_raw += 1
        else: w_raw += 1
            
    return int(j_raw * 3), int(w_raw * 1.5)

def get_status(count):
    if count <= 15: return "CLEAR"
    elif count <= 35: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    requests.post(url, json=payload)

if __name__ == "__main__":
    if download_traffic_image():
        johor, woodlands = analyze_traffic()
        weather = get_weather()
        j_status, w_status = get_status(johor), get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        csv_file = "data.csv"
        df_old = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()
        
        # Prepare and save new data
        new_row = pd.DataFrame([{"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Weather": weather}])
        df = pd.concat([df_old, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        # --- TELEGRAM NOTIFICATION LOGIC ---
        msg = (f"🚦 <b>Causeway Traffic Update</b> 🚦\n\n"
               f"🇲🇾 To Johor: {johor} ({j_status})\n"
               f"🇸🇬 To Woodlands: {woodlands} ({w_status})\n\n"
               f"🕒 {now} | {weather}\n"
               f"<a href='https://jamsniper.streamlit.app/'>View Live Cameras Here</a>")
        
        # 1. Check for Manual Trigger from GitHub Actions
        is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
        
        # 2. Check for Status Change
        status_changed = False
        if not df_old.empty:
            prev_j_status = get_status(df_old.iloc[-1]["To_Johor"])
            prev_w_status = get_status(df_old.iloc[-1]["To_Woodlands"])
            if j_status != prev_j_status or w_status != prev_w_status:
                status_changed = True
        else:
            status_changed = True # Send if it's the first time creating the CSV
            
        # Final Decision: Send if the status changed OR you triggered it manually
        if status_changed or is_manual:
            send_telegram(msg)