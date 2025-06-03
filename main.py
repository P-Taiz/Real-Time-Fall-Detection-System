import cv2
import time
import numpy as np
from ultralytics import YOLO
from fall_detector import PoseDetectorAdvanced, FallDetectorAdvanced
from telegram_alert import send_telegram_alert, reset_alert

def main():
    yolo_model = YOLO('weights/best.pt')
    pose_detector = PoseDetectorAdvanced(detectionCon=0.6, trackCon=0.6)
    fall_detector = FallDetectorAdvanced(history_size=10)
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    desired_width = 680
    desired_height = 440
    pTime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
    

        frame = cv2.resize(frame, (desired_width, desired_height))
        display_frame = frame.copy()
        results = yolo_model(frame, classes=0)

        is_fall = False
        current_bbox = None

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box = None
            best_conf = 0

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()

                if confidence > 0.7 and confidence > best_conf:
                    best_box = (x1, y1, x2, y2)
                    best_conf = confidence

            if best_box:
                x1, y1, x2, y2 = best_box
                current_bbox = best_box
                person_img = frame[y1:y2, x1:x2]

                if person_img.size > 0 and person_img.shape[0] > 10 and person_img.shape[1] > 10:
                    pose_detector.findPose(person_img)
                    lmList = pose_detector.findPosition(person_img, draw=True)

                    if len(lmList) > 0:
                        upper_height, leg_height, ratio = pose_detector.calculateBodyProportions()
                        body_angle = pose_detector.calculateBodyAngle()

                        if ratio is not None and body_angle is not None:
                            bbox_aspect_ratio = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 0

                            is_fall, fall_duration, confidence = fall_detector.update(
                                bbox_aspect_ratio, body_angle, time.time())

                            pose_detector.annotateActivity(display_frame, ratio, current_bbox)

                            cv2.putText(display_frame, f"Angle: {body_angle:.1f}", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            display_frame[y1:y2, x1:x2] = person_img

        if current_bbox:
            x1, y1, x2, y2 = current_bbox
            box_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 1)

        if is_fall:
            cv2.putText(display_frame, f"FALL DETECTED! {confidence:.2f}", (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if fall_duration and fall_duration > 3.5 and confidence >= 0.5:
                send_telegram_alert(display_frame)
        else:
            reset_alert()

        cTime = time.time()
        elapsed = cTime - pTime
        fps = 1 / elapsed if elapsed > 0 else 0
        pTime = cTime
        cv2.putText(display_frame, f"FPS: {int(fps)}", (5, 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv2.imshow('Fall Detection System', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()