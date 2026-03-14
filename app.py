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

# Motion count and last notification
motion_count = 0
last_notification_time = datetime.min

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
        if cv2.contourArea(contour) < 500:
            continue
        motion_detected = True
    return motion_detected

# --- Log motion function ---
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
        col1, col2 = st.columns([2, 1])
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam")
                break

            motion_detected = process_frame(frame)

            # Display original feed
            col1.subheader("Original Feed")
            col1.image(frame, channels="BGR")

            # Show motion detection rectangle only when motion detected
            processed_frame = frame.copy()
            if motion_detected:
                cv2.putText(processed_frame, "MOTION DETECTED", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            col1.subheader("Motion Detection")
            col1.image(processed_frame, channels="BGR")

            # Trigger alert every 2 minutes
            if motion_detected and datetime.now() - last_notification_time > timedelta(minutes=2):
                log_motion()
                st.sidebar.write(f"⚠️ Motion detected! Total: {motion_count}")
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)

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
            motion_detected = process_frame(frame)

            # Show original and motion detection once
            st.subheader("Original Image")
            st.image(frame, channels="BGR")

            st.subheader("Motion Detection")
            processed_frame = frame.copy()
            if motion_detected:
                cv2.putText(processed_frame, "MOTION DETECTED", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            st.image(processed_frame, channels="BGR")

            # Alert if needed
            if motion_detected and datetime.now() - last_notification_time > timedelta(minutes=2):
                log_motion()
                st.write(f"⚠️ Motion detected! Total: {motion_count}")
                filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)

        else:  # Video upload
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            # Show original video only once
            st.subheader("Original Video Uploaded")
            st.video(tfile)

            cap = cv2.VideoCapture(tfile)
            motion_detected_overall = False
            last_frame_with_motion = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                motion_detected = process_frame(frame)
                if motion_detected:
                    motion_detected_overall = True
                    last_frame_with_motion = frame.copy()

            cap.release()

            # Show final motion-detected frame only once
            if motion_detected_overall:
                processed_frame = last_frame_with_motion
                cv2.putText(processed_frame, "MOTION DETECTED", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                st.subheader("Motion Detected Snapshot")
                st.image(processed_frame, channels="BGR")

                # Trigger alert if 2 minutes passed
                if datetime.now() - last_notification_time > timedelta(minutes=2):
                    log_motion()
                    st.sidebar.write(f"⚠️ Motion detected! Total: {motion_count}")
                    filename = f"images/motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, processed_frame)
