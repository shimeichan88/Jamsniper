# JamSniper: AI Causeway Traffic Monitor

**JamSniper** is an automated AI traffic monitoring system for the Singapore-Johor Causeway. It uses computer vision (YOLOv8) to analyze real-time traffic camera feeds, split traffic counts by direction (To Johor vs. To Woodlands), and send alerts when heavy jams are detected.

## Features

* **AI Computer Vision:** Uses YOLOv8 to count vehicles (Cars, Trucks, Buses, Motorcycles) from LTA traffic cameras.
* **Split Logic:** Intelligently separates traffic into "To Johor" and "To Woodlands" lanes using geometric geometry.
* **Smart Weather:** Checks real-time rain sensors at the Causeway (with auto-backup to Woodlands Ave 9).
* **Live Dashboard:** A Streamlit web app displaying live charts, historical trends, and the latest analyzed image.
* **Quiet Mode:** The Telegram bot stays silent during light/moderate traffic and only alerts you when traffic is **Heavy (>50 cars)**.

## Live Dashboard
Link to my streamlit app here:  https://jamsniper.streamlit.app/

## How It Works

1.  **Data Collection:** A GitHub Action runs every 30 minutes.
2.  **Analysis:** The `bot.py` script downloads the latest LTA traffic image and runs object detection.
3.  **Alerting:**
    * If **Count > 50**: Sends a JAM Alert to Telegram.
    * If **Count < 50**: Stays silent.
4.  **Display:** The `traffic.py` script visualizes the collected data on the web dashboard.

## Configuration

### Thresholds
| Status | Count (Per Side) | Dashboard Color | Telegram Alert |
| :--- | :--- | :--- | :--- |
| **CLEAR** | 0 - 24 | Green | No |
| **MODERATE**| 25 - 50 | Yellow | No |
| **JAM** | 51+ | Red | **YES** |

### Environment Variables
To run this project, you need the following secrets set in your repository (GitHub/GitLab):

* `LTA_API_KEY`: API Key from LTA DataMall.
* `TELEGRAM_TOKEN`: Bot token from @BotFather.
* `TELEGRAM_CHAT_ID`: Your Telegram User/Group ID.

## Tech Stack

This project is built using the following technologies:

* **Python 3.9**: Core programming language.
* **Ultralytics YOLOv8**: AI model for object detection and vehicle counting.
* **Streamlit**: Framework for the web dashboard and visualization.
* **Pandas**: For data storage (CSV), manipulation, and analysis.
* **GitHub Actions**: Serverless automation to schedule the bot every 30 minutes.
* **OpenCV & Pillow**: For image processing and geometry calculations.
* **Requests**: For API calls to LTA DataMall and Singapore NEA (Weather).

