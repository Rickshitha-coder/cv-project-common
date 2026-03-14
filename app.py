# app.py
import streamlit as st
import cv2
import numpy as np
from datetime import datetime, timedelta
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
col1, col2 = st.columns([2, 1])
FRAME_WINDOW = col1.empty()
ORIGINAL_WINDOW = col1.empty()

# Log file path
log_file = "logs/motion_log.csv"

# Time of last notification (for 2-minute delay)
last_notification_time = datetime.min

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

# --- Function to log motion ---
def log_motion():
    global motion_count, last_notification_time
    motion_count += 1
    df = pd.DataFrame({"time": [datetime.now()], "motion_count": [motion_count]})
    df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
    last_notification_time = datetime.now()

# --- Webcam Mode ---
if mode == "Webcam (Local Only)":
    run = st.checkbox("Start Webcam (Local Only)")
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam")
                break

            processed_frame, motion_detected = process_frame(frame)

            # Display original frame
            ORIGINAL_WINDOW.subheader("Original Feed")
            ORIGINAL_WINDOW.image(frame, channels="BGR")

            # Display motion detection
            FRAME_WINDOW.subheader("Motion Detection")
            FRAME_WINDOW.image(processed_frame, channels="BGR")

            # Trigger notification only if 2 minutes have passed
            if motion_detected and datetime.now() - last_notification_time > timedelta(minutes=2):
                log_motion()
                st.sidebar.write(f"⚠️ Motion detected! Total: {motion_count}")
                # Save snapshot
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)

        cap.release()

# --- Upload Mode ---
else:
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","png","mp4"])
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_frame, motion_detected = process_frame(frame)

            # Show original and processed image
            col1.subheader("Original Image")
            col1.image(frame, channels="BGR")
            col1.subheader("Motion Detection")
            col1.image(processed_frame, channels="BGR")

            if motion_detected and datetime.now() - last_notification_time > timedelta(minutes=2):
                log_motion()
                st.write(f"⚠️ Motion detected! Total: {motion_count}")
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)

        else:  # Video upload
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, motion_detected = process_frame(frame)

                # Display original and processed frames
                col1.subheader("Original Video Feed")
                col1.image(frame, channels="BGR")
                col1.subheader("Motion Detection")
                col1.image(processed_frame, channels="BGR")

                # Trigger notification only every 2 minutes
                if motion_detected and datetime.now() - last_notification_time > timedelta(minutes=2):
                    log_motion()
                    st.write(f"⚠️ Motion detected! Total: {motion_count}")
                    filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, processed_frame)

            cap.release()
