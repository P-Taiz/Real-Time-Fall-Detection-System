import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
from fall_detector import PoseDetectorAdvanced, FallDetectorAdvanced
from telegram_alert import send_telegram_alert, reset_alert

# --- Minimal Streamlit display app centered ---
st.set_page_config(page_title="Fall Detection", layout="wide")
st.markdown("<h4 style='text-align:center;margin-bottom:0.5rem;'>Real-Time Fall Detection</h4>", unsafe_allow_html=True)

# Initialize video capture and models
cap = cv2.VideoCapture(0)
yolo_model = YOLO('weights/best.pt')
pose_detector = PoseDetectorAdvanced(detectionCon=0.6, trackCon=0.6)
fall_detector = FallDetectorAdvanced(history_size=10,
                                     fall_angle_threshold=45,
                                     fall_ratio_threshold=0.8)

# Place layout columns for centering
col1, col2, col3 = st.columns([1, 2, 1])
# Create an empty placeholder in center column
with col2:
    frame_window = st.empty()

# Main loop
p_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and copy for display
    frame = cv2.resize(frame, (680, 440))
    display = frame.copy()

    # YOLO person detection
    results = yolo_model(frame, classes=0)
    is_fall = False
    bbox = None

    if results and len(results[0].boxes) > 0:
        best = max(results[0].boxes, key=lambda x: x.conf[0].item())
        if best.conf[0].item() > 0.7:
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)
            person = frame[y1:y2, x1:x2]

            if person.size:
                pose_detector.findPose(person)
                lm = pose_detector.findPosition(person, draw=True)
                if lm:
                    ratio = pose_detector.calculateBodyProportions()[2]
                    angle = pose_detector.calculateBodyAngle()
                    aspect = (y2 - y1) / max(x2 - x1, 1)
                    is_fall, duration, conf = fall_detector.update(aspect, angle, time.time())
                    pose_detector.annotateActivity(display, ratio, bbox)
                    cv2.putText(display, f"Angle: {angle:.1f}", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    display[y1:y2, x1:x2] = person

    # Draw box and alert text
    if bbox:
        x1, y1, x2, y2 = bbox
        color = (0,0,255) if is_fall else (0,255,0)
        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
    if is_fall:
        cv2.putText(display,f"FALL DETECTED! {conf:.2f}", (200,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if duration and duration > 2 :
            send_telegram_alert(display)
    else:
        reset_alert()

    # FPS
    c_time = time.time()
    fps = 1/(c_time-p_time) if (c_time-p_time)>0 else 0
    p_time = c_time
    cv2.putText(display, f"FPS: {int(fps)}", (5,20),
                cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)

    # Display centered
    with col2:
        frame_window.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()
