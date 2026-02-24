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
IMG_PATH = "latest_traffic.jpg"

SHIFT = 0.28   
TILT = 0.43

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}
    requests.post(url, json=payload)

def get_weather():
    try:
        url = "https://api.data.gov.sg/v1/environment/rainfall"
        data = requests.get(url).json()
        stations = data['metadata']['stations']
        readings = data['items'][0]['readings']
        
        target_ids = ['S105', 'S104']
        rain_value = 0
        found = False
        
        for target_id in target_ids:
            for i, station in enumerate(stations):
                if station['id'] == target_id:
                    rain_value = readings[i]['value']
                    found = True
                    break
            if found: break
            
        if not found: return "☁️ Unknown"
        if rain_value == 0: return "☀️ Clear"
        elif rain_value < 5: return "🌧️ Light Rain"
        else: return "⛈️ Heavy Rain"
    except Exception:
        return "⚠️ Unavailable"

def analyze_traffic():
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": LTA_KEY, "accept": "application/json"}
    
    try:
        resp = requests.get(url, headers=headers).json()
        target_link = next((i['ImageLink'] for i in resp['value'] if str(i['CameraID']) == "2701"), None)
        if not target_link: return None, None
        
        img_data = requests.get(target_link).content
        with open(IMG_PATH, "wb") as f:
            f.write(img_data)
        
        img = Image.open(BytesIO(img_data))
        model = YOLO('yolov8m.pt')
        results = model(img, imgsz=1280, conf=0.05, iou=0.7, classes=[2, 3, 5, 7])
        
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
            
            if cy > (height * 0.60) and cx < (width * 0.30): continue 
            
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
        
        # 1. READ MEMORY (Check 30 mins ago)
        try: 
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                prev_cj = float(df.iloc[-1]["To_Johor"])
                prev_cw = float(df.iloc[-1]["To_Woodlands"])
            else:
                prev_cj, prev_cw = 0, 0
        except: 
            df = pd.DataFrame(columns=["Time", "Total_Count", "To_Johor", "To_Woodlands"])
            prev_cj, prev_cw = 0, 0
        
        # 2. SAVE NEW DATA
        new_row = {"Time": time_str, "Total_Count": cj+cw, "To_Johor": cj, "To_Woodlands": cw}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
        # 3. SMART NOTIFICATION LOGIC
        msg = ""
        
        # To Johor Logic
        if cj > 50 and prev_cj <= 50: 
            msg += f"🛑 **NEW JAM to Johor** (Congestion Index: {cj})\n"
        elif cj > 50 and cj >= (prev_cj + 15):
            msg += f"📈 **JAM WORSENING to Johor** (Congestion Index: {cj})\n"
        elif cj <= 20 and prev_cj > 20:
            msg += f"✅ **JAM CLEARED to Johor** (Congestion Index: {cj})\n"

        # To Woodlands Logic
        if cw > 50 and prev_cw <= 50: 
            msg += f"🛑 **NEW JAM to Woodlands** (Congestion Index: {cw})\n"
        elif cw > 50 and cw >= (prev_cw + 15):
            msg += f"📈 **JAM WORSENING to Woodlands** (Congestion Index: {cw})\n"
        elif cw <= 20 and prev_cw > 20:
            msg += f"✅ **JAM CLEARED to Woodlands** (Congestion Index: {cw})\n"
            
        # 4. SEND ALERT IF NEEDED
        if msg:
            weather = get_weather()
            dashboard_url = "https://jamsniper.streamlit.app/" 
            full_msg = f"🚦 **Traffic Alert** ({time_str})\n\n{msg}\n**Weather:** {weather}\n\n📺 [View Live Cameras Here]({dashboard_url})"
            send_telegram(full_msg)