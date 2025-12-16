import cv2
import mediapipe as mp
import time
import pygame

pygame.mixer.init()

def warning():
    pygame.mixer.music.load("D:\\Came_dashboard\\siren-a-248662.mp3")
    pygame.mixer.music.play()
    time.sleep(4)
    pygame.mixer.music.stop()

LINE1_START = (x, y)
LINE1_END = (x, y)
LINE2_START = (x, y)
LINE2_END = (x, y)

#example:
# LINE1_START = (100, 200)
# LINE1_END = (400, 200)
# LINE2_START = (100, 400)
# LINE2_END = (400, 400)

mp_pose = mp.solutions.pose

prev_right_leg = None
prev_left_leg = None
crossed_flag = False
crossed_time = 0

def cross_val(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def is_crossed(prev_point, curr_point, p1, p2):
    if not prev_point or not curr_point:
        return False
    x1, y1 = p1
    x2, y2 = p2
    prev_side = cross_val(prev_point[0], prev_point[1], x1, y1, x2, y2)
    curr_side = cross_val(curr_point[0], curr_point[1], x1, y1, x2, y2)
    return prev_side * curr_side < 0

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        alert = False
        msg = ""

        if results.pose_landmarks:
            try:
                right_toe = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                left_toe = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            except IndexError:
                continue

            curr_right = (int(right_toe.x * w), int(right_toe.y * h))
            curr_left = (int(left_toe.x * w), int(left_toe.y * h))

            crossed = False
            if prev_right_leg is not None:
                if (is_crossed(prev_right_leg, curr_right, LINE1_START, LINE1_END) or
                    is_crossed(prev_right_leg, curr_right, LINE2_START, LINE2_END)):
                    crossed = True
            if prev_left_leg is not None:
                if (is_crossed(prev_left_leg, curr_left, LINE1_START, LINE1_END) or
                    is_crossed(prev_left_leg, curr_left, LINE2_START, LINE2_END)):
                    crossed = True

            if crossed:
                crossed_flag = True
                crossed_time = time.time()
                warning()

            prev_right_leg = curr_right
            prev_left_leg = curr_left

        if crossed_flag:
            if time.time() - crossed_time < 1.5:
                alert = True
                msg = "Leg crossed a line!"
            else:
                crossed_flag = False

        cv2.line(frame, LINE1_START, LINE1_END, (0, 255, 255), 3)
        cv2.line(frame, LINE2_START, LINE2_END, (255, 0, 255), 3)

        if alert:
            cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Live Line-Cross Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
