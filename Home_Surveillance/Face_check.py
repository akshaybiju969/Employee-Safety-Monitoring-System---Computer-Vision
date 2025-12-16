import cv2
import mediapipe as mp
import time
import pygame
import os

pygame.mixer.init()

def warning():
    pygame.mixer.music.load("D:\\AI cam2\\siren-a-248662.mp3")
    pygame.mixer.music.play()
    time.sleep(4)
    pygame.mixer.music.stop()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('D:/INNODOTS CAM/haarcascade_mcs_mouth.xml')

cap = cv2.VideoCapture(0)
cv2.namedWindow("Suspicious Behavior Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Suspicious Behavior Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

P1 = 0
p1_start_time = None
p1_displayed = False
probability_display_start_time = None
last_time_A1_activated = None

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

            if abs(left_ear.x - right_ear.x) < 0.05:
                text = "Facing Back"
                P1 = 0
                p1_start_time = None
                p1_displayed = False
            else:
                text = "Facing Front"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(20, 20))
                mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=15, minSize=(30, 15))

                B1 = 1 if len(eyes) > 0 else 0
                B2 = 1 if len(mouths) > 0 else 0

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

                if (B1 == 1 and B2 == 0) or (B1 == 0 and B2 == 1) or (B1 == 0 and B2 == 0):
                    if P1 == 0:
                        p1_start_time = time.time()
                        p1_displayed = False
                    P1 = 1
                else:
                    P1 = 0
                    p1_start_time = None
                    p1_displayed = False

                if P1 == 1 and p1_start_time and not p1_displayed:
                    if time.time() - p1_start_time >= 2:
                        probability_display_start_time = time.time()
                        p1_displayed = True

            if probability_display_start_time:
                if time.time() - probability_display_start_time <= 3:
                    cv2.putText(frame, "Probability = 50%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    probability_display_start_time = None

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            P1 = 0
            p1_start_time = None
            p1_displayed = False

        if P1 > 0:
            if last_time_A1_activated is None:
                last_time_A1_activated = time.time()
            elif time.time() - last_time_A1_activated > 17:
                warning()
                last_time_A1_activated = time.time()

        cv2.imshow('Suspicious Behavior Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
