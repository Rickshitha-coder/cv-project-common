import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="Motion Detection & Alert System", layout="wide")
st.title("🛡️ Motion Detection & Alert System")

mode = st.radio("Select Mode:", ["Webcam (Local Only)", "Upload Video/Image (Cloud Compatible)"])

FRAME_WINDOW = st.image([])

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
min_contour_area = 500
motion_count = 0

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    fgmask = fgbg.apply(gray)
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame, motion_detected

if mode == "Webcam (Local Only)":
    run = st.checkbox("Start Webcam (Local Only)")
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam")
                break
            frame, motion_detected = process_frame(frame)
            if motion_detected:
                motion_count += 1
                st.write(f"⚠️ Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            FRAME_WINDOW.image(frame, channels="BGR")
        cap.release()

else:  # Upload mode for Cloud
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","png","mp4"])
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame, motion_detected = process_frame(frame)
            if motion_detected:
                motion_count += 1
                st.write(f"⚠️ Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            FRAME_WINDOW.image(frame, channels="BGR")
        else:  # Video
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
                    st.write(f"⚠️ Motion detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                FRAME_WINDOW.image(frame, channels="BGR")
            cap.release()
