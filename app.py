import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Motion Detection & Alert System", layout="wide")
st.title("🛡️ Real-Time Motion Detection & Alert System")

# Webcam capture checkbox
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Motion detection parameters
min_contour_area = 500  # Minimum area of detected contour
motion_count = 0

# Main loop
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to read from webcam")
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate to fill gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    if motion_detected:
        motion_count += 1
        st.write(f"⚠️ Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()