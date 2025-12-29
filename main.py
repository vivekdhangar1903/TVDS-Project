import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")

model = YOLO("yolov8n.pt")

stop_line_y = 400
violation_duration = 5

def helmet_check(frame, person_box):
    x1, y1, x2, y2 = map(int, person_box)

    w = x2 - x1
    h = y2 - y1

    hx1 = x1 + int(w * 0.3)
    hx2 = x1 + int(w * 0.7)
    hy1 = y1
    hy2 = y1 + int(h * 0.25)

    head = frame[hy1:hy2, hx1:hx2]

    if head.size == 0:
        return False

    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])

    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    dark_ratio = np.sum(mask > 0) / mask.size

    if dark_ratio > 0.4:
        return True

    return False

def overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)


def is_red_signal(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
           cv2.inRange(hsv, lower_red2, upper_red2)

    return cv2.countNonZero(mask) > 200

def is_yellow_signal(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 120, 120])
    upper_yellow = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return cv2.countNonZero(mask) > 200

def run_images():
    for img_name in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (800, 600))
        results = model(frame)

        boxes = results[0].boxes
        if boxes is not None:
            for box, cls in zip(boxes.xyxy, boxes.cls):
                if int(cls) == 0:
                    helmet = helmet_check(frame, box)
                    x1, y1, x2, y2 = map(int, box)

                    if helmet:
                        color = (0, 255, 0)
                        label = "HELMET"
                    else:
                        color = (0, 0, 255)
                        label = "NO HELMET"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("TVDS Image Mode", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def run_video():
    video_files = [v for v in os.listdir(VIDEO_FOLDER)
                   if v.lower().endswith((".mp4", ".avi", ".mov"))]

    for video_name in video_files:
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_name))
        red_memory = {}
        yellow_memory = {}
        direction_memory = {}


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))
            results = model.track(frame, persist=True)

            signal_state = "green"

            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls) == 9:
                    if is_red_signal(frame, box):
                        signal_state = "red"
                    elif is_yellow_signal(frame, box):
                        signal_state = "yellow"

            cv2.line(frame, (0, stop_line_y), (800, stop_line_y), (0, 0, 255), 2)
            current_time = time.time()

            boxes = results[0].boxes
            if boxes.id is None:
                cv2.imshow("TVDS Video Mode", frame)
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break
                continue

            for box, cls, tid in zip(boxes.xyxy, boxes.cls, boxes.id):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(tid)

                if int(cls) in [2, 3, 5, 7]:
                    if signal_state == "red" and y2 > stop_line_y:
                        red_memory.setdefault(track_id, current_time)

                    if signal_state == "yellow" and y2 > stop_line_y:
                        yellow_memory.setdefault(track_id, current_time)

                    center_x = int((x1 + x2) / 2)

                    if track_id not in direction_memory:
                        direction_memory[track_id] = center_x
                    else:
                        prev_x = direction_memory[track_id]

                        if center_x < prev_x - 5:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            cv2.putText(frame,
                                        f"WRONG DIRECTION ID {track_id}",
                                        (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (255, 0, 0),
                                        2)

                        direction_memory[track_id] = center_x

                if track_id in red_memory and current_time - red_memory[track_id] <= violation_duration:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"RED VIOLATION ID {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue

                if track_id in yellow_memory and current_time - yellow_memory[track_id] <= violation_duration:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.putText(frame, f"YELLOW CROSS ID {track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    continue

                if int(cls) == 0:
                    helmet = helmet_check(frame, box)
                    color = (0, 255, 0) if helmet else (0, 0, 255)
                    label = "HELMET" if helmet else "NO HELMET"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"{signal_state.upper()} SIGNAL", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if signal_state == "red" else (0, 165, 255), 3)

            cv2.putText(frame, f"VIDEO: {video_name}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("TVDS Video Mode", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

        cap.release()

    cv2.destroyAllWindows()

if config.mode == "image":
    run_images()
elif config.mode == "video":
    run_video()
