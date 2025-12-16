from ultralytics import YOLO
import cv2
import cvzone
import numpy as np


FRAME_WIDTH = 960
FRAME_HEIGHT = 720
PANEL_WIDTH = 350
PANEL_BG_COLOR = (30, 30, 30)

HELMET_CLASS_NAME = "helmetw"
PPE_CLASS_NAME = "ppekit"


def load_models():
    try:
        model_people = YOLO("yolov8n.pt")
        model_helmet = YOLO("helmet_best.pt")
        model_ppe = YOLO("ppe_kit.pt")
        print("Models loaded successfully")
        return model_people, model_helmet, model_ppe
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")


def overlap(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)


def get_detections(frame, model_people, model_helmet, model_ppe):
    try:
        people_result = model_people(frame, conf=0.25, iou=0.45, imgsz=960, verbose=False)[0]
        helmet_result = model_helmet(frame, conf=0.12, iou=0.35, imgsz=960, verbose=False)[0]
        ppe_result = model_ppe(frame, conf=0.12, iou=0.35, imgsz=960, verbose=False)[0]

        person_boxes = []
        for b in people_result.boxes:
            if model_people.names.get(int(b.cls[0])) == "person" and float(b.conf[0]) > 0.20:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                person_boxes.append((x1, y1, x2, y2))

        helmet_boxes = []
        for b in helmet_result.boxes:
            if model_helmet.names.get(int(b.cls[0])) == HELMET_CLASS_NAME:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                helmet_boxes.append((x1, y1, x2, y2))

        ppe_boxes = []
        for b in ppe_result.boxes:
            if model_ppe.names.get(int(b.cls[0])) == PPE_CLASS_NAME:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                ppe_boxes.append((x1, y1, x2, y2))

        return person_boxes, helmet_boxes, ppe_boxes

    except Exception as e:
        print("Detection error:", e)
        return [], [], []


def classify_person(person_box, helmet_boxes, ppe_boxes):
    has_helmet = any(overlap(person_box, hb) for hb in helmet_boxes)
    has_ppe = any(overlap(person_box, pb) for pb in ppe_boxes)

    if has_helmet and has_ppe:
        return "Helmet + PPE", (0, 255, 0), "both"
    if has_helmet:
        return "Helmet Only", (255, 165, 0), "helmet"
    if has_ppe:
        return "PPE Only", (0, 255, 255), "ppe"
    return "NO SAFETY!", (0, 0, 255), "none"


def draw_panel(total, helmet_only, ppe_only, both, no_safety):
    panel = np.zeros((FRAME_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = PANEL_BG_COLOR

    cv2.putText(panel, "ANALYTICS PANEL", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    stats = [
        ("Total Persons", total),
        ("Helmet Only", helmet_only),
        ("PPE Only", ppe_only),
        ("Both", both),
        ("No Safety", no_safety)
    ]

    y = 80
    for label, value in stats:
        cv2.putText(panel, f"{label}: {value}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 50

    return panel


def main():
    model_people, model_helmet, model_ppe = load_models()

    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Camera not detected or cannot be opened")

    print("Camera started")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            person_boxes, helmet_boxes, ppe_boxes = get_detections(
                frame, model_people, model_helmet, model_ppe
            )

            total = len(person_boxes)
            helmet_count = ppe_count = both = no_safety = 0

            for person_box in person_boxes:
                label, color, category = classify_person(person_box, helmet_boxes, ppe_boxes)

                if category == "both":
                    both += 1
                elif category == "helmet":
                    helmet_count += 1
                elif category == "ppe":
                    ppe_count += 1
                else:
                    no_safety += 1

                x1, y1, x2, y2 = person_box
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), colorC=color)
                cvzone.putTextRect(frame, label, (x1, y1 - 10),
                                   scale=1, thickness=2, colorT=color)

            panel = draw_panel(total, helmet_count, ppe_count, both, no_safety)
            combined = np.hstack((frame, panel))

            cv2.imshow("Smart PPE & Helmet Monitoring System", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print("Runtime error:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")


if __name__ == "__main__":
    main()
