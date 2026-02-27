import os
import requests
import pandas as pd
from datetime import datetime
import pytz
from ultralytics import YOLO

# --- SECURE CREDENTIALS ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def get_weather():
    try:
        # Fetches current temperature for the Woodlands/Causeway area
        url = "https://api.open-meteo.com/v1/forecast?latitude=1.4481&longitude=103.7757&current_weather=true"
        response = requests.get(url).json()
        return f"{response['current_weather']['temperature']}°C"
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
            if cam['CameraID'] == "2701": # Woodlands Causeway camera
                img_data = requests.get(cam['ImageLink']).content
                with open("latest_traffic.jpg", "wb") as f:
                    f.write(img_data)
                return True
    except:
        return False
    return False

def analyze_traffic():
    model = YOLO("yolov8n.pt") 
    # Use HD (1280) and hide labels for a professional, clean look
    results = model("latest_traffic.jpg", conf=0.10, iou=0.5, classes=[2, 3, 5, 7], imgsz=1280)
    results[0].save("latest_traffic.jpg", labels=False) 
    
    j_raw, w_raw = 0, 0
    img_width = results[0].orig_shape[1]
    divider = img_width * (0.5 + 0.28) # Your 78% mark logic
    
    for box in results[0].boxes:
        center_x = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        if center_x > divider:
            j_raw += 1
        else:
            w_raw += 1
            
    # Apply "Causeway Perspective" multipliers to estimate total volume accurately
    return int(j_raw * 3), int(w_raw * 1.5)

def get_status(count):
    # Calibrated thresholds based on the new multiplier logic
    if count < 12: return "CLEAR"
    elif count < 25: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

if __name__ == "__main__":
    if download_traffic_image():
        johor, woodlands = analyze_traffic()
        weather = get_weather()
        
        j_status = get_status(johor)
        w_status = get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        # This line handles the creation of a new CSV if you deleted the old one
        csv_file = "data.csv"
        df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame(columns=["Time", "To_Johor", "To_Woodlands", "Weather"])
        
        new_row = {"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Weather": weather}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        msg = (f"🚦 Causeway Traffic Update 🚦\n\n"
               f"🇲🇾 To Johor: {johor} ({j_status})\n"
               f"🇸🇬 To Woodlands: {woodlands} ({w_status})\n\n"
               f"🕒 {now} | 🌡️ {weather}")
        
        # Logic to only alert you when the traffic level changes
        if len(df) > 1:
            prev_j = get_status(df.iloc[-2]["To_Johor"])
            prev_w = get_status(df.iloc[-2]["To_Woodlands"])
            if j_status != prev_j or w_status != prev_w:
                send_telegram(msg)