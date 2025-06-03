import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque

class PoseDetectorAdvanced:
    def __init__(self, static_mode=False, model_complexity=1, smooth=True, 
                 detectionCon=0.5, trackCon=0.5):
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.results = None
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                img, 
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    
    def getLandmarks(self):
        return self.results.pose_landmarks if self.results else None
    
    def calculateDistance(self, point1, point2):
        if not point1 or not point2:
            return 0
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def calculateBodyProportions(self):
        landmarks = self.getLandmarks()
        if landmarks is None:
            return None, None, None

        left_upper = self.calculateDistance(
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_upper = self.calculateDistance(
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP])
        upper_height = (left_upper + right_upper) / 2

        left_leg = self.calculateDistance(
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE])
        right_leg = self.calculateDistance(
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE])
        leg_height = (left_leg + right_leg) / 2

        ratio = leg_height / upper_height if upper_height > 0 else 0
        return upper_height, leg_height, ratio
    
    def calculateBoundingBox(self, img):
        landmarks = self.getLandmarks()
        if landmarks is None:
            return None
            
        h, w, _ = img.shape
        x_min, y_min, x_max, y_max = w, h, 0, 0

        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        padding = int(0.05 * w)
        return (max(0, x_min - padding), 
                max(0, y_min - padding),
                min(w, x_max + padding), 
                min(h, y_max + padding))
    
    def calculateBodyAngle(self):
        landmarks = self.getLandmarks()
        if landmarks is None:
            return None
            
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
    
            body_vector = [nose.x - hip_center_x, nose.y - hip_center_y]

            angle = np.degrees(np.arctan2(body_vector[0], -body_vector[1]))
            return abs(angle)
        except:
            return None
    
    def annotateActivity(self, img, ratio, bbox=None):
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.putText(img, f"Ratio: {ratio:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


class FallDetectorAdvanced:
    def __init__(self, history_size=10, fall_angle_threshold=50, fall_ratio_threshold=0.8):
        self.fall_detected = False
        self.fall_timestamp = None
        self.fall_duration = 0
        self.aspect_ratio_history = deque(maxlen=history_size)
        self.angle_history = deque(maxlen=history_size)
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_ratio_threshold = fall_ratio_threshold
        self.confidence = 0 

    def update(self, aspect_ratio, body_angle, timestamp):
        if aspect_ratio is None or body_angle is None:
            return False, None, 0
      
        if aspect_ratio is not None:
            self.aspect_ratio_history.append(aspect_ratio)
        if body_angle is not None:
            self.angle_history.append(body_angle)
      
        if len(self.aspect_ratio_history) < 3 or len(self.angle_history) < 3:
            return False, None, 0

        smooth_ratio = np.median(self.aspect_ratio_history)
        smooth_angle = np.median(self.angle_history)

        angle_condition = smooth_angle > self.fall_angle_threshold
        ratio_condition = smooth_ratio < self.fall_ratio_threshold
        
        fall_condition = (angle_condition and smooth_ratio < 1.2) or ratio_condition

        angle_weight = 0.6
        ratio_weight = 0.4
        self.confidence = 0
        
        if angle_condition:
            self.confidence += angle_weight * (smooth_angle / 90)
        if ratio_condition:
            self.confidence += ratio_weight * (1 - smooth_ratio/self.fall_ratio_threshold)
  
        if fall_condition:
            if not self.fall_detected:
                self.fall_detected = True
                self.fall_timestamp = timestamp
            self.fall_duration = timestamp - self.fall_timestamp
            return True, self.fall_duration, self.confidence
        else:
            self.fall_detected = False
            self.fall_timestamp = None
            self.fall_duration = 0
            return False, None, 0
