import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pygame


pygame.mixer.init()
def warning():
    pygame.mixer.music.load("D:\\AI cam2\\siren-a-248662.mp3")
    pygame.mixer.music.play()
    time.sleep(4)
    pygame.mixer.music.stop()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

LINE_START = (x, y)
LINE_END = (x, y)
#example-
# LINE_START = (300, 50)
# LINE_END = (300, 400)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def distance_point_to_line(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return num / (den + 1e-6)

prev_left_wrist = None
prev_right_wrist = None
prev_time = time.time()
angle_below_90_start = None
WARNING_TIME = 5
warning_end_time = None

cap = cv2.VideoCapture(0)
cv2.namedWindow("Combined Hand Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Combined Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()
        dt = current_time - prev_time
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(image_rgb)
        pose_results = pose.process(image_rgb)
        prev_time = current_time
        cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3)
        left_speed = right_speed = 0

        if holistic_results.left_hand_landmarks:
            wrist = holistic_results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
            lx, ly = int(wrist.x * w), int(wrist.y * h)
            if prev_left_wrist:
                dx, dy = lx - prev_left_wrist[0], ly - prev_left_wrist[1]
                left_speed = math.sqrt(dx**2 + dy**2) / dt
            prev_left_wrist = (lx, ly)

        if holistic_results.right_hand_landmarks:
            wrist = holistic_results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
            rx, ry = int(wrist.x * w), int(wrist.y * h)
            if prev_right_wrist:
                dx, dy = rx - prev_right_wrist[0], ry - prev_right_wrist[1]
                right_speed = math.sqrt(dx**2 + dy**2) / dt
            prev_right_wrist = (rx, ry)

        cv2.putText(frame, f"Left Hand Speed: {left_speed:.2f} px/s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Right Hand Speed: {right_speed:.2f} px/s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if left_speed > 2340 or right_speed >2340:
            print("⚠️ High hand speed detected — Triggering Warning")
            warning_end_time = time.time() + 3
            warning()

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            r_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))
            r_elbow = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
            r_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))
            l_elbow = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
            l_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))

            for point in [r_wrist, r_elbow, r_shoulder, l_wrist, l_elbow, l_shoulder]:
                cv2.circle(frame, point, 6, (255,255,0), -1)

            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            cv2.putText(frame, f"Right Arm Angle: {int(r_angle)}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Left Arm Angle: {int(l_angle)}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            right_condition = distance_point_to_line(r_wrist, LINE_START, LINE_END) <= 50 and r_angle <= 145
            left_condition = distance_point_to_line(l_wrist, LINE_START, LINE_END) <= 50 and l_angle <= 145

            if right_condition or left_condition:
                if angle_below_90_start is None:
                    angle_below_90_start = time.time()
                else:
                    elapsed = time.time() - angle_below_90_start
                    if elapsed >= WARNING_TIME:
                        warning_end_time = time.time() + 3
                        angle_below_90_start = None
                        warning()
            else:
                angle_below_90_start = None

        if warning_end_time and time.time() <= warning_end_time:
            cv2.putText(frame, "⚠️ WARNING DETECTED!", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
        elif warning_end_time and time.time() > warning_end_time:
            warning_end_time = None

        cv2.imshow("Combined Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
