# JamSniper: AI Causeway Traffic Monitor

## 1. What is JamSniper: AI Causeway Traffic Monitor
JamSniper is an automated traffic monitoring system that uses computer vision to analyze real-time traffic conditions on the Singapore-Johor Causeway. The project consists of a background bot (`bot.py`) that captures and analyzes camera feeds, and an interactive Streamlit dashboard (`traffic.py`) that visualizes the current traffic flow heading towards Johor (Malaysia) and Woodlands (Singapore).

## 2. Features
* **AI-Powered Vehicle Counting:** Utilizes an extra-large YOLOv8 model (`yolov8x.pt`) at a high resolution to detect and count vehicles, even from a distance.
* **Real-Time Data Ingestion:** Automatically downloads live traffic images from the LTA DataMall API (specifically Camera ID 2701).
* **Smart Telegram Alerts:** Sends notifications to a specified Telegram chat only when the overall traffic status changes, preventing notification spam.
* **Live Weather Integration:** Fetches real-time weather data using the Open-Meteo API. It pulls the current temperature and a specific weather code, mapping the code to human-readable conditions (e.g., Clear, Drizzle, Heavy Rain) to provide a combined status like "31.0°C | Heavy Rain".
* **Interactive Dashboard:** Features a Streamlit interface displaying the live annotated image, 24-hour traffic trend line charts, and horizontal bar charts for current traffic distribution.

## 3. How It Works
1. **Image Capture:** The bot fetches the latest image of the Causeway from the LTA DataMall API and saves it locally.
2. **Detection & Separation:** The YOLOv8 model scans the image to identify bounding boxes for vehicles. A mathematically calibrated dividing line separates the road into the "To Johor" lane and the "To Woodlands" lane based on bounding box coordinates.
3. **Distance Weighting & Safety Cap:** Because vehicles further away appear smaller, the algorithm applies a perspective factor based on the bounding box's vertical position and area. A safety cap is applied to restrict any single bounding box from counting as more than 35 vehicles to prevent massive glitch errors.
4. **Weather Context:** The `get_weather()` function queries the Open-Meteo API for the exact coordinates of the Causeway. It extracts the current temperature and translates the numerical weather code into a descriptive condition string.
5. **Data Persistence:** The calculated vehicle counts, along with the timestamp (in Asia/Singapore timezone) and the formatted weather data, are appended to a persistent `data.csv` file.
6. **Visualization:** The Streamlit dashboard reads `data.csv` to calculate traffic status and updates the live metrics and charts for end-users.

## 4. Configuration (Including  Algorithm, Calculations and Thresholds)
The system uses several configurable parameters and thresholds to categorize traffic severity:

* **Line Calibration:** The dividing line separating the two directions of traffic can be adjusted via the Streamlit dashboard sidebar. The default coordinates are set to Top (X: 1.0, Y: 0.31) and Bottom (X: 0.35, Y: 0.93).
 * **Traffic Analysis Algorithm & Calculations:** Because camera feeds look straight down a long road, a single bounding box far away in the horizon might represent a cluster of 20 tightly packed cars, while a bounding box close up represents just 1 car. JamSniper uses mathematical weighting to estimate true traffic density:
    1. **High-Fidelity Scanning:** The bot uses the Extra-Large YOLOv8 model (`yolov8x.pt`) and scales the image up to a 1280px resolution (`imgsz=1280`) specifically to detect tiny, distant cars near the horizon.
    2. **Lane Sorting:** Every detected vehicle's center point `(cx, cy)` is compared against the dynamic divider line on the screen. If the vehicle is to the left of the line, it is counted towards Johor; if on the right, it is counted towards Woodlands.
    3. **Perspective Factor & Distance Weighting:** * The algorithm calculates a **Perspective Factor (`norm_h`)** determining how far "up" the image the vehicle is (0.0 is the bottom, 1.0 is the horizon).
       * It then calculates a weight based on the vehicle's bounding box area (`box_area`).
       * **The Math:** `weight = 1.2 + (850 / (box_area + 25)) * (norm_h ** 1.5)`. This effectively means a bounding box with a small area that is situated high up near the horizon is weighted much heavier than a large bounding box at the bottom of the image. This is tuned so that 8-10 distant detections accurately estimate roughly 200 physical cars.
    4. **The Safety Cap:** To prevent computer vision glitches (like mistakenly detecting a massive billboard as a distant car cluster), a hard limit is imposed: `min(weight, 35.0)`. No single bounding box is ever allowed to count for more than 35 cars, ensuring the total data remains stable and accurate.
* **Bot Alert Thresholds:** The background alert system categorizes traffic based on vehicle counts:
  * **CLEAR:** Less than 55 vehicles.
  * **MODERATE:** Between 55 and 149 vehicles.
  * **JAM:** 150 vehicles or more.
* **Dashboard Display Thresholds:** The front-end Streamlit dashboard displays color-coded statuses using slightly different margins:
  * **CLEAR (Green):** Less than 61 vehicles.
  * **MODERATE (Yellow):** Between 61 and 160 vehicles.
  * **JAM (Red):** 161 vehicles or more.

## 5. Environment Variables
To run the bot securely, the following environment variables must be configured in your deployment environment:
* `LTA_API_KEY`: Your Account Key for the LTA DataMall API.
* `TELEGRAM_TOKEN`: The API token for your Telegram Bot.
* `TELEGRAM_CHAT_ID`: The target Chat ID where the bot will send status alerts.

## 6. Tech Stack
* **Frontend:** Streamlit
* **AI & Computer Vision:** Ultralytics (YOLOv8), OpenCV (`opencv-python` and `opencv-python-headless`), Pillow
* **Data Handling & Processing:** Pandas, Requests, Pytz
* **System Dependencies:** `libgl1-mesa-glx` (required for OpenCV image processing operations)

## 7. Data Flow Architecture
The JamSniper ecosystem relies on a seamless pipeline between the code repository, the application runtime, and the user notification system:

* **GitHub (Source & Deployment Trigger):** GitHub acts as the central version control repository, storing all application files (`bot.py`, `traffic.py`, `requirements.txt`, `packages.txt`, etc.). It serves as the source of truth for the codebase. When new code or configuration changes are pushed to the GitHub repository, it acts as a trigger for continuous deployment.
* **Streamlit Community Cloud (Hosting & Execution):** Streamlit is directly linked to the GitHub repository. It automatically pulls the latest code from GitHub to build, host, and serve the live web dashboard. Streamlit provides the active runtime environment where the Python code executes, the YOLOv8 model runs its computer vision tasks on the downloaded traffic images, and the resulting data (`data.csv`) is visualized for users.
* **Telegram (Real-Time Alerting):** Telegram acts as the outbound notification endpoint. While the heavy lifting (image processing and logic) happens in the Streamlit/hosting environment, the system monitors for state changes (e.g., traffic going from "CLEAR" to "JAM"). When a change is detected, the Python script sends an HTTP POST request containing the calculated vehicle counts, weather, and a link to the dashboard directly to the Telegram Bot API, delivering a real-time push notification to the configured chat.

```mermaid
graph TD
    %% Nodes
    GitHub["fa:fa-github GitHub<br/>(Source & Deployment Trigger)"]
    Streamlit["fa:fa-server Streamlit Community Cloud<br/>(Hosting & Execution)"]
    LTA["fa:fa-camera LTA DataMall API<br/>(Live Traffic Images)"]
    OpenMeteo["fa:fa-cloud Open-Meteo API<br/>(Live Weather Data)"]
    Telegram["fa:fa-telegram Telegram API<br/>(Real-Time Alerts)"]
    Users["fa:fa-users End Users"]

    %% Flow
    GitHub -- "Pulls Codebase on Push<br/>(bot.py, traffic.py)" --> Streamlit
    LTA -- "Downloads Image<br/>(Camera 2701)" --> Streamlit
    OpenMeteo -- "Fetches Temp & Condition" --> Streamlit
    
    subgraph Streamlit Processing
        Streamlit -- "YOLOv8 Analysis & Weather" --> Data["data.csv<br/>(Vehicle Counts, Weather & Status)"]
    end
    
    Data -- "State Change Detected<br/>(e.g., CLEAR to JAM)" --> Telegram
    Data -- "Visualized on Dashboard" --> Users
    Telegram -- "Push Notification<br/>with Dashboard Link" --> Users