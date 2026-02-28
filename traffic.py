import streamlit as st
import pandas as pd
import os
import cv2

# 1. PAGE SETUP
st.set_page_config(page_title="JamSniper", layout="centered")
st.title("🚦 JamSniper: Causeway Traffic")

# --- SIDEBAR: CALIBRATION ---
st.sidebar.title("📏 Line Calibration")
tx = st.sidebar.slider("Top X", 0.0, 1.0, 1.0)
ty = st.sidebar.slider("Top Y", 0.0, 1.0, 0.31)
bx = st.sidebar.slider("Bottom X", 0.0, 1.0, 0.35)
by = st.sidebar.slider("Bottom Y", 0.0, 1.0, 0.93)

# 2. LIVE IMAGE DISPLAY
st.write("---")
if os.path.exists("latest_traffic.jpg"):
    img = cv2.imread("latest_traffic.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    start, end = (int(w * tx), int(h * ty)), (int(w * bx), int(h * by))
    # Draw the divider line on the dashboard
    cv2.line(img, start, end, (0, 255, 0), 10)
    st.image(img, caption="Live View: Johor (Left) | Woodlands (Right)", use_container_width=True)
else:
    st.info("Waiting for the first traffic image to be processed...")

# 3. DATA DISPLAY (SYNCED WITH BOT.PY)
try:
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        if not df.empty:
            latest = df.iloc[-1]
            
            # Pull the calculated values from the CSV
            val_j = int(latest["To_Johor"])
            val_w = int(latest["To_Woodlands"])

            st.write(f"**Last Update:** {latest['Time']}")

            col1, col2 = st.columns(2)
            
            # --- To Johor Direction ---
            with col1:
                st.metric("To Johor 🇲🇾", val_j)
                if val_j < 61: 
                    st.success("✅ CLEAR")
                elif val_j < 161: 
                    st.warning("⚠️ MODERATE") 
                else: 
                    st.error("🛑 JAM")
                
            # --- To Woodlands Direction ---
            with col2:
                st.metric("To Woodlands 🇸🇬", val_w)
                if val_w < 61: 
                    st.success("✅ CLEAR")
                elif val_w < 161: 
                    st.warning("⚠️ MODERATE") 
                else: 
                    st.error("🛑 JAM")

            # 4. TREND CHARTS
            st.write("---")
            st.subheader("📈 24-Hour Traffic Trend")
            df['Time'] = pd.to_datetime(df['Time'])
            # Show last 48 entries (approx 24 hours if updating every 30 mins)
            chart_data = df.tail(48).copy()
            chart_data["Display_Time"] = chart_data["Time"].dt.strftime("%H:%M")
            st.line_chart(chart_data.set_index("Display_Time")[["To_Johor", "To_Woodlands"]])

            # 5. BUSINESS PROBLEM 1: CURRENT DISTRIBUTION
            st.write("---")
            st.subheader("📊 Current Traffic Distribution")
            # Best chart for Sub-business problem 1: Horizontal Bar Chart
            dist_df = pd.DataFrame({
                'Direction': ['To Johor', 'To Woodlands'],
                'Vehicles': [val_j, val_w]
            })
            st.bar_chart(dist_df.set_index('Direction'), horizontal=True)

    else:
        st.error("data.csv not found. Please run the bot first to generate data.")

except Exception as e:
    st.error(f"Error loading data: {e}")

# REFRESH BUTTON
if st.button("Manual Refresh"):
    st.rerun()