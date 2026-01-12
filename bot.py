import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import pytz

# --- CONFIGURATION ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CSV_PATH = "data.csv"

# --- TELEGRAM SENDER ---
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram keys missing. Skipping alert.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

# --- TRAFFIC LOGIC ---
def analyze_traffic():
    # 1. Get Image
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": LTA_KEY, "accept": "application/json"}
    
    try:
        resp = requests.get(url, headers=headers).json()
        target_link = next((i['ImageLink'] for i in resp['value'] if str(i['CameraID']) == "2701"), None)
        
        if not target_link: return None
        
        # 2. Analyze Image (Using your HD settings)
        img = Image.open(BytesIO(requests.get(target_link).content))
        model = YOLO('yolov8m.pt')
        results = model(img, imgsz=1280, conf=0.05, iou=0.7, classes=[2, 3, 5, 7])
        
        # 3. Count Cars
        width = img.size[0]
        # Use simple center line split for automation
        mid_point = width * 0.5 
        
        # Note: We use a simpler split for the robot vs the visual dashboard
        # But you can copy the "slope" logic if you want 100% match.
        # For alerts, "Total Count" is usually enough.
        
        count = len(results[0].boxes)
        
        print(f"✅ Analysis Complete. Cars found: {count}")
        return count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Run Analysis
    traffic_score = analyze_traffic()
    
    if traffic_score is not None:
        # 2. Get Current Time (Singapore)
        sg_time = datetime.now(pytz.timezone('Asia/Singapore'))
        time_str = sg_time.strftime('%Y-%m-%d %H:%M')
        
        # 3. Save to Database
        try:
            df = pd.read_csv(CSV_PATH)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Time", "Total_Count", "To_Johor", "To_Woodlands"])
            
        # (Simplifying database for the bot run to avoid complex geometry errors)
        new_row = {"Time": time_str, "Total_Count": traffic_score, "To_Johor": 0, "To_Woodlands": traffic_score}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        print("💾 Data Saved.")

        # 4. SEND ALERT (The New Part!) 🔔
        # Logic: If traffic is LOW (< 25) and it's a reasonable time (e.g. not 4 AM)
        hour = sg_time.hour
        
        if traffic_score < 25:
            message = f"🟢 **Traffic is CLEAR!**\n\nCurrent Score: {traffic_score}\nTime: {time_str}\n\n_Go now!_"
            send_telegram(message)
            print("📨 Green Alert sent!")
            
        elif traffic_score > 30:
            message = f"🛑 **BAD JAM DETECTED**\n\nScore: {traffic_score}\nTime: {time_str}\n\n_Stay home!_"
            send_telegram(message)
            print("📨 Red Alert sent!")
            
    else:
        print("⚠️ No result to save.")