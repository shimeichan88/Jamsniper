import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime
import pytz
import cv2

# --- CONFIGURATION ---
LTA_KEY = os.environ.get("LTA_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CSV_PATH = "data.csv"
IMG_PATH = "latest_traffic.jpg"  # <--- WE WILL SAVE THE IMAGE NOW

# YOUR GEOMETRY (ADJUST THESE IF JOHOR IS ALWAYS 0)
SHIFT = 0.28   # If Johor is 0, try increasing this to 0.35
TILT = 0.43

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def analyze_traffic():
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": LTA_KEY, "accept": "application/json"}
    
    try:
        # 1. Get Image
        resp = requests.get(url, headers=headers).json()
        target_link = next((i['ImageLink'] for i in resp['value'] if str(i['CameraID']) == "2701"), None)
        if not target_link: return None, None
        
        # 2. Save Image for Website
        img_data = requests.get(target_link).content
        with open(IMG_PATH, "wb") as f:
            f.write(img_data)
        
        # 3. Analyze
        img = Image.open(BytesIO(img_data))
        model = YOLO('yolov8m.pt')
        results = model(img, imgsz=1280, conf=0.05, iou=0.7, classes=[2, 3, 5, 7])
        
        # 4. Geometry Split
        width, height = img.size
        base_top = width * 0.60
        base_bottom = width * 0.40
        top_x = base_top + (width * SHIFT) + (width * TILT)
        bottom_x = base_bottom + (width * SHIFT) - (width * TILT)
        slope = (bottom_x - top_x) / height
        
        j_count = 0
        w_count = 0
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            
            # Billboard Filter
            if cy > (height * 0.60) and cx < (width * 0.30): continue 
            
            # Divide
            divider_x = top_x + (slope * cy)
            if cx < divider_x: j_count += 1
            else: w_count += 1
                
        return j_count, w_count
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    cj, cw = analyze_traffic()
    
    if cj is not None:
        sg_time = datetime.now(pytz.timezone('Asia/Singapore'))
        time_str = sg_time.strftime('%Y-%m-%d %H:%M')
        
        # Save Data
        try: df = pd.read_csv(CSV_PATH)
        except: df = pd.DataFrame(columns=["Time", "Total_Count", "To_Johor", "To_Woodlands"])
        
        new_row = {"Time": time_str, "Total_Count": cj+cw, "To_Johor": cj, "To_Woodlands": cw}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
        # SEND SPLIT ALERTS 🔔
        msg = ""
        # Alert for Johor
        if cj < 25: msg += f"🟢 **Johor is CLEAR ({cj})**\n"
        elif cj > 45: msg += f"🛑 **JAM to Johor ({cj})**\n"
        
        # Alert for Woodlands
        if cw < 25: msg += f"🟢 **Woodlands is CLEAR ({cw})**\n"
        elif cw > 45: msg += f"🛑 **JAM to Woodlands ({cw})**\n"
        
        if msg:
            full_msg = f"🚦 **Traffic Update** ({time_str})\n\n{msg}\n_Drive safe!_"
            send_telegram(full_msg)