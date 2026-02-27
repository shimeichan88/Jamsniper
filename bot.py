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

# --- 1. FETCH IMAGE FROM LTA DATAMALL ---
def download_traffic_image():
    headers = {'AccountKey': LTA_KEY, 'accept': 'application/json'}
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return False
        data = response.json()
        
        for cam in data['value']:
            if cam['CameraID'] == "2701":
                img_url = cam['ImageLink']
                img_data = requests.get(img_url).content
                with open("latest_traffic.jpg", "wb") as f:
                    f.write(img_data)
                return True
    except Exception as e:
        print(f"Failed to download image: {e}")
    return False

# --- 2. ANALYZE TRAFFIC WITH RELIABILITY FIXES ---
def analyze_traffic():
    model = YOLO("yolov8n.pt") 
    
    # imgsz=1280 for HD eyes, conf=0.10 to catch distant vehicles
    results = model("latest_traffic.jpg", conf=0.10, iou=0.5, classes=[2, 3, 5, 7], imgsz=1280)
    
    # labels=False hides the percentage numbers for a cleaner look
    results[0].save("latest_traffic.jpg", labels=False) 
    
    johor_raw = 0
    woodlands_raw = 0
    johor_buses = 0
    woodlands_buses = 0
    
    img_width = results[0].orig_shape[1]
    divider = img_width * (0.5 + 0.28) # Your 78% divider line
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        
        # Determine Direction first
        is_johor = center_x > divider
        
        # Count Buses/Trucks (classes 5 and 7) by direction
        if cls in [5, 7]:
            if is_johor: johor_buses += 1
            else: woodlands_buses += 1
            
        # Count Cars (classes 2 and 3) by direction
        if is_johor:
            johor_raw += 1
        else:
            woodlands_raw += 1
            
    # Apply Multipliers for perspective
    johor_total = int(johor_raw * 3)
    woodlands_total = int(woodlands_raw * 1.5)
            
    return johor_total, woodlands_total, johor_buses, woodlands_buses

# --- 3. UPDATED THRESHOLDS FOR RELIABILITY ---
def get_status(count):
    # Because we use multipliers, a count of 25 means the bridge is physically full
    if count < 12: return "CLEAR"
    elif count < 25: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if download_traffic_image():
        johor, woodlands, buses = analyze_traffic()
        
        johor_status = get_status(johor)
        woodlands_status = get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        
        csv_file = "data.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time", "To_Johor", "To_Woodlands", "Total", "Buses"])
            
        new_row = {"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Total": johor + woodlands, "Buses": buses}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        
        # Determine if we need to send an alert
        if len(df) > 1:
            prev_johor = get_status(df.iloc[-2]["To_Johor"])
            prev_woodlands = get_status(df.iloc[-2]["To_Woodlands"])
        else:
            prev_johor, prev_woodlands = "", ""
            
        if johor_status != prev_johor or woodlands_status != prev_woodlands:
            msg = (f"🚦 Causeway Traffic Update 🚦\n\n"
                   f"🇲🇾 To Johor: {johor} ({johor_status})\n"
                   f"🇸🇬 To Woodlands: {woodlands} ({woodlands_status})\n"
                   f"🚌 Buses/Trucks Detected: {buses}\n\n"
                   f"🕒 {now}")
            send_telegram(msg)