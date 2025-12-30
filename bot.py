import os
import csv
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_KEY = os.environ.get("LTA_API_KEY")

# ⚠️ IMPORTANT: If you changed the sliders on your Dashboard, update these numbers!
SHIFT = 0.05
TILT = 0.25

def count_cars():
    # 1. Fetch Image
    url = "https://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2"
    headers = {"AccountKey": API_KEY, "accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers).json()
        target_link = None
        for img in resp['value']:
            if str(img['CameraID']) == "2701":
                target_link = img['ImageLink']
                break

        if not target_link:
            print("Error: Camera 2701 not found")
            return None, None

        # 2. Analyze Image
        img_resp = requests.get(target_link)
        img = Image.open(BytesIO(img_resp.content))

        model = YOLO('yolov8m.pt')
        results = model(img, imgsz=1024, conf=0.15, iou=0.6, classes=[2, 3, 5, 7])[0]

        # 3. Math Logic
        width, height = img.size
        base_top = width * 0.60
        base_bottom = width * 0.40
        top_x = base_top + (width * SHIFT) + (width * TILT)
        bottom_x = base_bottom + (width * SHIFT) - (width * TILT)
        slope = (bottom_x - top_x) / height

        to_johor = 0
        to_woodlands = 0

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_w = x2 - x1
            box_h = y2 - y1

            # Filters (Billboard & Zone)
            if center_y > (height * 0.60) and center_x < (width * 0.30): continue
            if (box_w / box_h) > 3.0: continue

            # Counting
            divider_x = top_x + (slope * center_y)
            if center_x < divider_x:
                to_johor += 1
            else:
                to_woodlands += 1

        return to_johor, to_woodlands

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    j, w = count_cars()
    if j is not None:
        # Time in Singapore (UTC+8)
        sg_time = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
        print(f"Update: {sg_time} | Johor: {j} | Woodlands: {w}")

        # Save to CSV
        with open("data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sg_time, j, w])
