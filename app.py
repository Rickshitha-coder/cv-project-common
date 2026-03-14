# app.py
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import os
import pandas as pd

# --- Setup ---
st.set_page_config(page_title="🛡️ Motion Detection & Alert System", layout="wide")
st.title("🛡️ Motion Detection & Alert System")

# Create folders if not exist
os.makedirs("images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mode selection
mode = st.radio("Select Mode:", ["Webcam (Local Only)", "Upload Video/Image"])

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Motion count
motion_count = 0

# Streamlit display placeholder
FRAME_WINDOW = st.image([])

# Log file path
log_file = "logs/motion_log.csv"

# --- Helper function ---
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # fixed minimum area
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame, motion_detected

# --- Webcam Mode ---
if mode == "Webcam (Local Only)":
    run = st.checkbox("Start Webcam (Local Only)")
    if run:
        cap = cv2.VideoCapture(0)
        col1, col2 = st.columns([2, 1])
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam")
                break

            frame, motion_detected = process_frame(frame)

            if motion_detected:
                motion_count += 1
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                st.sidebar.write(f"⚠️ Motion detected! Total: {motion_count}")

                # Append to log CSV
                df = pd.DataFrame({"time": [datetime.now()], "motion_count": [motion_count]})
                df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))

            # Display frames
            col1.subheader("Live Feed")
            col1.image(frame, channels="BGR")

            # Motion log
            col2.subheader("Motion Log")
            if os.path.exists(log_file):
                logs_df = pd.read_csv(log_file)
                col2.dataframe(logs_df.tail(10))

        cap.release()

# --- Upload Mode ---
else:
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","png","mp4"])
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame, motion_detected = process_frame(frame)

            if motion_detected:
                motion_count += 1
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                st.write(f"⚠️ Motion detected! Total: {motion_count}")
                df = pd.DataFrame({"time": [datetime.now()], "motion_count": [motion_count]})
                df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))

            st.image(frame, channels="BGR")

        else:  # Video upload
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, motion_detected = process_frame(frame)
                if motion_detected:
                    motion_count += 1
                    filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    st.write(f"⚠️ Motion detected! Total: {motion_count}")
                    df = pd.DataFrame({"time": [datetime.now()], "motion_count": [motion_count]})
                    df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
                st.image(frame, channels="BGR")
            cap.release()
