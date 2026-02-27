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

def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        response = requests.get(url).json()
        temp = response['current_weather']['temperature']
        code = response['current_weather']['weathercode']
        
        # Mapping weather codes to icons
        rain_status = ""
        if code in [51, 53, 55]: rain_status = " | 🌧️ Drizzle"
        elif code in [61, 63, 65, 80, 81]: rain_status = " | ⛈️ Rain"
        elif code in [66, 67, 82]: rain_status = " | 🌊 Heavy Rain"
            
        return f"{temp}°C{rain_status}"
    except:
        return "N/A"

def download_traffic_image():
    headers = {'AccountKey': LTA_KEY, 'accept': 'application/json'}
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return False
        data = response.json()
        for cam in data['value']:
            if cam['CameraID'] == "2701":
                img_data = requests.get(cam['ImageLink']).content
                with open("latest_traffic.jpg", "wb") as f:
                    f.write(img_data)
                return True
    except:
        return False
    return False

def analyze_traffic():
    model = YOLO("yolov8n.pt") 
    # Using your preferred sensitivity (0.10) and HD (1280)
    results = model("latest_traffic.jpg", conf=0.10, iou=0.5, classes=[2, 3, 5, 7], imgsz=1280)
    
    # Save base AI detections
    results[0].save("latest_traffic.jpg", labels=False) 
    
    # Load image for diagonal line drawing
    img = cv2.imread("latest_traffic.jpg")
    h, w, _ = img.shape
    
    # SETTINGS: 78% at top, 45% at bottom to match road perspective
    top_x = int(w * 0.78)
    bottom_x = int(w * 0.45)
    
    # Draw green diagonal line
    cv2.line(img, (top_x, 0), (bottom_x, h), (0, 255, 0), 5)
    cv2.imwrite("latest_traffic.jpg", img)
    
    j_raw, w_raw = 0, 0
    for box in results[0].boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        
        # Geometry: Is the car center right of the diagonal line?
        line_x_at_y = bottom_x + (top_x - bottom_x) * (cy / h)
        
        if cx > line_x_at_y:
            j_raw += 1
        else:
            w_raw += 1
            
    # Multipliers: 3x for Johor, 1.5x for Woodlands
    return int(j_raw * 3), int(w_raw * 1.5)

def get_status(count):
    if count <= 15: return "CLEAR"
    elif count <= 35: return "MODERATE"
    else: return "JAM"

def send_telegram(message, image_path):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    # Send message with weather
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url_msg, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    # Send processed photo with green line
    url_img = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as f:
        requests.post(url_img, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f})

if __name__ == "__main__":
    if download_traffic_image():
        johor, woodlands = analyze_traffic()
        weather = get_weather()
        
        j_status = get_status(johor)
        w_status = get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        csv_file = "data.csv"
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame(columns=["Time", "To_Johor", "To_Woodlands", "Weather"])
        new_row = {"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Weather": weather}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        msg = (f"🚦 Causeway Traffic Update 🚦\n\n"
               f"🇲🇾 To Johor: {johor} ({j_status})\n"
               f"🇸🇬 To Woodlands: {woodlands} ({w_status})\n\n"
               f"🕒 {now} | 🌡️ {weather}")
        
        # Notify if status changes
        if len(df) > 1:
            prev_j = get_status(df.iloc[-2]["To_Johor"])
            prev_w = get_status(df.iloc[-2]["To_Woodlands"])
            if j_status != prev_j or w_status != prev_w:
                send_telegram(msg, "latest_traffic.jpg")