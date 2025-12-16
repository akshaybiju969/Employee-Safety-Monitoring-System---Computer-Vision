import streamlit as st
import cv2
import base64
import numpy as np
import face_recognition
from ultralytics import YOLO
import cvzone
import os
import time


def setup_page():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    .stAlert {display: none;}
    main {padding: 0 !important;}
    main, .stApp { background-color: #333 !important; }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-18ni7ap {
        background-color: #333 !important;
    }
    </style>
    """, unsafe_allow_html=True)


MODEL_PATH = "../YOLO-Weights/yolov8n.pt"
FACES_FOLDER = "peoples"

PROCESS_SCALE = 0.5
FACE_EVERY_N_FRAMES = 5
FPS_SLEEP = 0.03

PANEL_WIDTH = 800
PANEL_HEIGHT = 1200
PANEL_BG = "#888"
PANEL_PADDING = 100

FRAME_WIDTH = 1460
FRAME_HEIGHT = 1220
TOP_PADDING = 10
LEFT_PADDING = 100


def load_yolo_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()


def load_known_faces(folder):
    encodings = []
    names = []

    if not os.path.isdir(folder):
        st.warning(f"Folder '{folder}' not found")
        return encodings, names

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                path = os.path.join(folder, filename)
                img = face_recognition.load_image_file(path)
                face_enc = face_recognition.face_encodings(img)

                if face_enc:
                    encodings.append(face_enc[0])
                    name = os.path.splitext(filename)[0]
                    name = "".join([c for c in name if not c.isdigit()])
                    names.append(name.capitalize())
            except Exception as e:
                st.warning(f"Failed to load face {filename}: {e}")

    return encodings, names


def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    return cap


def process_faces(frame, frame_id, known_encodings, known_names):
    detected_names = []
    face_boxes = []

    if frame_id % FACE_EVERY_N_FRAMES != 0:
        return detected_names, face_boxes

    try:
        small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for (top, right, bottom, left), face_enc in zip(locations, encodings):
            name = "Unknown"

            if known_encodings:
                matches = face_recognition.compare_faces(
                    known_encodings, face_enc, tolerance=0.5
                )
                distances = face_recognition.face_distance(
                    known_encodings, face_enc
                )
                best_match = np.argmin(distances)

                if matches[best_match]:
                    name = known_names[best_match]

            detected_names.append(name)

            top_a = int(top / PROCESS_SCALE)
            right_a = int(right / PROCESS_SCALE)
            bottom_a = int(bottom / PROCESS_SCALE)
            left_a = int(left / PROCESS_SCALE)

            face_boxes.append((left_a, top_a, right_a, bottom_a, name))

    except Exception as e:
        st.warning(f"Face recognition error: {e}")

    return detected_names, face_boxes


def process_yolo(frame, model):
    boxes = []
    person_count = 0

    try:
        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if label == "person":
                    person_count += 1

                boxes.append((x1, y1, x2, y2, label, conf))

    except Exception as e:
        st.warning(f"YOLO inference error: {e}")

    return boxes, person_count


def draw_detections(frame, yolo_boxes, face_boxes):
    for x1, y1, x2, y2, label, conf in yolo_boxes:
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), rt=1, t=2)
        cvzone.putTextRect(
            frame, f"{label} {conf:.2f}", (x1, max(30, y1)), scale=1
        )

    for l, t, r, b, name in face_boxes:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (l, t), (r, b), color, 2)


def render_panel(container, detected_faces, people_count, chance):
    if not detected_faces:
        faces_html = "<p style='color:white; font-size:32px;'>No face detected</p>"
    else:
        faces_html = "<p style='color:white; font-size:34px; font-weight:bold;'>Faces:</p><ul>"
        for n in set(detected_faces):
            faces_html += f"<li style='color:yellow; font-size:30px;'>{n}</li>"
        faces_html += "</ul>"

    people_html = f"""
    <p style='color:white; font-size:34px; font-weight:bold;'>
        People count: <span style='color:lime; font-size:38px;'>{people_count}</span>
    </p>
    <p style='color:white; font-size:34px; font-weight:bold;'>
        Stampede Chance: <span style='color:lime; font-size:38px;'>{chance}%</span>
    </p>
    """

    container.markdown(
        f"""
        <div style="
        width:{PANEL_WIDTH}px;
        height:{PANEL_HEIGHT}px;
        background:{PANEL_BG};
        padding:{PANEL_PADDING}px;
        border-radius:10px;
        color:white;
        overflow:auto;">
            <h2 style="font-size:36px;">Recognition Panel</h2>
            {faces_html}
            <hr style="border:0; height:2px; background:#555;">
            {people_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_video(container, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (FRAME_WIDTH, FRAME_HEIGHT))
    _, buffer = cv2.imencode(".jpg", frame_rgb)
    data = base64.b64encode(buffer).decode("utf-8")

    container.markdown(
        f"""
        <div style="padding-top:{TOP_PADDING}px; padding-left:{LEFT_PADDING}px;">
            <img src="data:image/jpg;base64,{data}" 
            style="width:{FRAME_WIDTH}px; height:{FRAME_HEIGHT}px; border-radius:8px;">
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    setup_page()

    model = load_yolo_model()
    known_encodings, known_names = load_known_faces(FACES_FOLDER)

    col_left, col_right = st.columns([3, 1])
    video_container = col_left.empty()
    panel_container = col_right.empty()

    cap = init_camera()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found")
            break

        frame_id += 1

        faces, face_boxes = process_faces(
            frame, frame_id, known_encodings, known_names
        )
        yolo_boxes, people_count = process_yolo(frame, model)

        chance = min(100, round((people_count / 13) * 100))

        draw_detections(frame, yolo_boxes, face_boxes)
        render_panel(panel_container, faces, people_count, chance)
        render_video(video_container, frame)

        time.sleep(FPS_SLEEP)

    cap.release()


if __name__ == "__main__":
    main()
