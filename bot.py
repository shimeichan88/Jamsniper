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
        print(f"LTA Server Status Code: {response.status_code}")
        if response.status_code != 200: print(f"Server Message: {response.text[:200]}")
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

# --- 2. YOLOv8 INFERENCE (THE ROLLBACK + OCCLUSION UPGRADE) ---
def analyze_traffic():
    model = YOLO("yolov8n.pt") 
    
    results = model("latest_traffic.jpg", conf=0.10, iou=0.5, classes=[2, 3, 5, 7], imgsz=1280)
    
    johor_count = 0
    woodlands_count = 0
    
    boxes = results[0].boxes
    img_width = results[0].orig_shape[1]
    
    SHIFT = 0.28 
    divider = img_width * (0.5 + SHIFT)
    
    for box in boxes:
        # We are back to simple counting: 1 Box = 1 Vehicle!
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        
        if center_x > divider:
            johor_count += 1
        else:
            woodlands_count += 1
            
    return johor_count, woodlands_count

# --- 3. STATE MACHINE & THRESHOLDS ---
def get_status(count):
    if count < 25: return "CLEAR"
    elif count < 50: return "MODERATE"
    else: return "JAM"

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting JamSniper Bot...")
    
    if download_traffic_image():
        print("Image downloaded successfully. Running YOLOv8 AI...")
        johor, woodlands = analyze_traffic()
        
        johor_status = get_status(johor)
        woodlands_status = get_status(woodlands)
        
        sgt = pytz.timezone('Asia/Singapore')
        
        # --- THE FIXES ARE RIGHT HERE ---
        now = datetime.now(sgt).strftime("%Y-%m-%d %H:%M") 
        total_cars = johor + woodlands 
        # --------------------------------
        
        csv_file = "data.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time", "To_Johor", "To_Woodlands", "Total"])
            
        new_row = {"Time": now, "To_Johor": johor, "To_Woodlands": woodlands, "Total": total_cars}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"Data saved: {johor} to Johor, {woodlands} to Woodlands. Total: {total_cars}")
        
        if len(df) > 1:
            prev_johor = get_status(df.iloc[-2]["To_Johor"])
            prev_woodlands = get_status(df.iloc[-2]["To_Woodlands"])
        else:
            prev_johor, prev_woodlands = "", ""
            
        if johor_status != prev_johor or woodlands_status != prev_woodlands:
            msg = (f"🚦 Causeway Traffic Update 🚦\n\n"
                   f"🇲🇾 To Johor: {johor} ({johor_status})\n"
                   f"🇸🇬 To Woodlands: {woodlands} ({woodlands_status})\n\n"
                   f"🕒 {now}")
            send_telegram(msg)
            print("Telegram alert sent!")
        else:
            print("No status change. Skipping Telegram alert.")
    else:
        print("Failed to run pipeline: Could not download image.")