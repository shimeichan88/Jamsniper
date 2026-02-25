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
        
        j_area_total = 0
        w_area_total = 0
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            
            if cy > (height * 0.60) and cx < (width * 0.30): continue 
            
            # --- NEW UPGRADE: CALCULATE AREA INSTEAD OF COUNTING ---
            box_area = (x2 - x1) * (y2 - y1)
            
            divider_x = top_x + (slope * cy)
            if cx < divider_x: j_area_total += box_area
            else: w_area_total += box_area
                
        # Divide by 1000 to keep the index numbers small and readable
        return int(j_area_total / 1000), int(w_area_total / 1000)
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_traffic_status(index):
    """Helper function to determine the current traffic state."""
    if index < 40: return "CLEAR"
    elif index < 80: return "MODERATE"
    else: return "JAM"

if __name__ == "__main__":
    cj, cw = analyze_traffic()
    
    if cj is not None:
        sg_time = datetime.now(pytz.timezone('Asia/Singapore'))
        time_str = sg_time.strftime('%Y-%m-%d %H:%M')
        
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
            
        new_row = {"Time": time_str, "Total_Count": cj+cw, "To_Johor": cj, "To_Woodlands": cw}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        
        # --- NEW UPGRADE: STATE MACHINE LOGIC ---
        msg = ""
        
        # Get Previous and Current States
        prev_johor_status = get_traffic_status(prev_cj)
        curr_johor_status = get_traffic_status(cj)
        
        prev_wood_status = get_traffic_status(prev_cw)
        curr_wood_status = get_traffic_status(cw)
        
        # To Johor Alert Logic (Only alert if the STATUS changes)
        if curr_johor_status != prev_johor_status:
            if curr_johor_status == "JAM":
                msg += f"🛑 **JAM to Johor** (Density Index: {cj})\n"
            elif curr_johor_status == "MODERATE":
                msg += f"⚠️ **Moderate Traffic to Johor** (Density Index: {cj})\n"
            elif curr_johor_status == "CLEAR":
                msg += f"✅ **Cleared to Johor** (Density Index: {cj})\n"

        # To Woodlands Alert Logic (Only alert if the STATUS changes)
        if curr_wood_status != prev_wood_status:
            if curr_wood_status == "JAM":
                msg += f"🛑 **JAM to Woodlands** (Density Index: {cw})\n"
            elif curr_wood_status == "MODERATE":
                msg += f"⚠️ **Moderate Traffic to Woodlands** (Density Index: {cw})\n"
            elif curr_wood_status == "CLEAR":
                msg += f"✅ **Cleared to Woodlands** (Density Index: {cw})\n"
            
        # Send Alert
        if msg:
            weather = get_weather()
            dashboard_url = "https://jamsniper.streamlit.app/" 
            full_msg = f"🚦 **Traffic Alert** ({time_str})\n\n{msg}\n**Weather:** {weather}\n\n📺 [View Live Cameras Here]({dashboard_url})"
            send_telegram(full_msg)