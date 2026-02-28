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

# --- MANUAL COORDINATES
TX = 1.0   # Top X (Horizontal)
TY = 0.31  # Top Y (Vertical - moved down from top)
BX = 0.35  # Bottom X (Horizontal)
BY = 0.93  # Bottom Y (Vertical - moved up from bottom)

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
    results = model("latest_traffic.jpg", conf=CONFIDENCE, iou=0.5, classes=[2, 3, 5, 7], imgsz=1280)
    
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    top_x, bottom_x = int(w * SHIFT_TOP), int(w * SHIFT_BOTTOM)
    
    # Draw green line for your web view
    cv2.line(img, (top_x, 0), (bottom_x, h), (0, 255, 0), 5)
    cv2.imwrite("latest_traffic.jpg", img)
    
    j_raw, w_raw = 0, 0
    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        line_x = bottom_x + (top_x - bottom_x) * (cy / h)
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
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame(columns=["Time", "To_Johor", "To_Woodlands", "Weather"])
        new_row = {"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Weather": weather}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        # Exact format with clickable Hyperlink
        msg = (f"🚦 <b>Causeway Traffic Update</b> 🚦\n\n"
               f"🇲🇾 To Johor: {johor} ({j_status})\n"
               f"🇸🇬 To Woodlands: {woodlands} ({w_status})\n\n"
               f"🕒 {now} | {weather}\n"
               f"<a href='https://jamsniper.streamlit.app/'>View Live Cameras Here</a>")
        
        if len(df) > 1:
            if j_status != get_status(df.iloc[-2]["To_Johor"]) or w_status != get_status(df.iloc[-2]["To_Woodlands"]):
                send_telegram(msg)