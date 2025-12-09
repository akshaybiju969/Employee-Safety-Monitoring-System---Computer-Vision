from ultralytics import YOLO
import cv2
import cvzone
import numpy as np


model_people = YOLO("yolov8n.pt")
model_helmet = YOLO("helmet_best.pt")
model_ppe = YOLO("ppe_kit.pt")


helmet_class = "helmetw"
ppe_class = "ppekit"

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

panel_width = 350
panel_bg_color = (30, 30, 30)

def overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    results_people = model_people(frame, conf=0.25, iou=0.45, imgsz=960, verbose=False)[0]
    results_helmet = model_helmet(frame, conf=0.12, iou=0.35, imgsz=960, verbose=False)[0]
    results_ppe = model_ppe(frame, conf=0.12, iou=0.35, imgsz=960, verbose=False)[0]

    person_boxes = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]))
        for b in results_people.boxes
        if model_people.names.get(int(b.cls[0])) == "person" and float(b.conf[0]) > 0.20
    ]

    helmet_boxes = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]))
        for b in results_helmet.boxes
        if model_helmet.names.get(int(b.cls[0])) == helmet_class and float(b.conf[0]) > 0.10
    ]

    ppe_boxes = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]))
        for b in results_ppe.boxes
        if model_ppe.names.get(int(b.cls[0])) == ppe_class and float(b.conf[0]) > 0.10
    ]

    total_persons = len(person_boxes)
    helmet_only = ppe_only = both_ppe_helmet = no_safety = 0

    for (px1, py1, px2, py2) in person_boxes:
        has_helmet = any(overlap((px1, py1, px2, py2), hb) for hb in helmet_boxes)
        has_ppe = any(overlap((px1, py1, px2, py2), pb) for pb in ppe_boxes)

        if has_helmet and has_ppe:
            both_ppe_helmet += 1
            label, color = "Helmet + PPE", (0, 255, 0)
        elif has_helmet:
            helmet_only += 1
            label, color = "Helmet Only", (255, 165, 0)
        elif has_ppe:
            ppe_only += 1
            label, color = "PPE Only", (0, 255, 255)
        else:
            no_safety += 1
            label, color = "NO SAFETY!", (0, 0, 255)

        cvzone.cornerRect(frame, (px1, py1, px2 - px1, py2 - py1), colorC=color)
        cvzone.putTextRect(frame, label, (px1, py1 - 10), scale=1, thickness=2, colorT=color)

    panel = np.zeros((FRAME_HEIGHT, panel_width, 3), dtype=np.uint8)
    panel[:] = panel_bg_color

    texts = [
        ("Total Persons", total_persons),
        ("Helmet Only", helmet_only),
        ("PPE Only", ppe_only),
        ("Both", both_ppe_helmet),
        ("No Safety", no_safety)
    ]

    cv2.putText(panel, "ANALYTICS PANEL", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    y = 80
    for label, val in texts:
        cv2.putText(panel, f"{label}: {val}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y += 50

    combined = np.hstack((frame, panel))
    cv2.imshow("Smart PPE & Helmet Monitoring System", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
