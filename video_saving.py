import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import datetime
import os
from ultralytics import YOLO


class OptimizedHandMonitor:
    def __init__(self):
        # ---------------- AOI Polygon ----------------
        self.AOI_POLYGON = [(200, 100), (600, 100), (600, 400), (200, 400)]
        self.drawing_mode = False
        self.temp_polygon = []

        # ---------------- Frame Management ----------------
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_skip = 2
        self.frame_count = 0

        # ---------------- Zone Tracking ----------------
        self.current_zone = None
        self.last_zone_message_time = 0
        self.zone_message_cooldown = 1.0

        # ---------------- Intrusion Saving ----------------
        self.intrusion_save_path = "intrusions"
        os.makedirs(self.intrusion_save_path, exist_ok=True)
        self.last_intrusion_save_time = 0
        self.intrusion_save_cooldown = 1.0  # seconds

        # ---------------- Mediapipe (optional visualization) ----------------
        # self.mp_hands = mp.solutions.hands
        # self.mp_draw = mp.solutions.drawing_utils
        # self.hands = self.mp_hands.Hands(static_image_mode=False,
        #                                  max_num_hands=2,
        #                                  min_detection_confidence=0.7,
        #                                  min_tracking_confidence=0.5)

        # ---------------- YOLO Model ----------------
        self.yolo_model = YOLO(r"runs\detect\train\weights\best.pt")

        # Compile AOI zones
        self.update_compiled_polygon()

    def update_compiled_polygon(self):
        """Compile AOI and create yellow/red zones"""
        self.compiled_polygon = np.array(self.AOI_POLYGON, dtype=np.int32)

        x_coords = [p[0] for p in self.AOI_POLYGON]
        y_coords = [p[1] for p in self.AOI_POLYGON]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bbox_height = max_y - min_y
        split_y = min_y + int(bbox_height * 0.2)

        self.yellow_zone_polygon = [
            (min_x, min_y), (max_x, min_y),
            (max_x, split_y), (min_x, split_y)
        ]
        self.red_zone_polygon = [
            (min_x, split_y), (max_x, split_y),
            (max_x, max_y), (min_x, max_y)
        ]

        self.compiled_yellow_zone = np.array(self.yellow_zone_polygon, np.int32)
        self.compiled_red_zone = np.array(self.red_zone_polygon, np.int32)

    def draw_polygon(self, event, x, y, flags, param):
        """Mouse-based AOI drawing"""
        if self.drawing_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_polygon.append((x, y))
                print(f"[INFO] Added point: {(x, y)}")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.temp_polygon) >= 3:
                    self.AOI_POLYGON = self.temp_polygon.copy()
                    self.update_compiled_polygon()
                    print(f"[INFO] AOI Polygon updated: {self.AOI_POLYGON}")
                    self.temp_polygon = []
                    self.drawing_mode = False
                else:
                    print("[WARNING] Need at least 3 points to finalize polygon.")

    def point_in_poly_fast(self, pt, polygon):
        return cv2.pointPolygonTest(polygon, pt, False) >= 0

    def get_hand_zone(self, pt):
        if self.point_in_poly_fast(pt, self.compiled_red_zone):
            return "red"
        elif self.point_in_poly_fast(pt, self.compiled_yellow_zone):
            return "yellow"
        return None

    def frame_capture_thread(self, cap):
        """Captures frames in a background thread"""
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass

    def process_yolo_detections(self, img):
        """Detect glove/hand using YOLO; ignore background"""
        yolo_results = self.yolo_model(img, verbose=False)
        hand_detected = False
        current_zone = None

        for r in yolo_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
                if conf < 0.5:
                    continue

                class_name = self.yolo_model.names[int(cls_id)].lower()

                # Ignore background detections
                if "background" in class_name:
                    continue

                if "glove" in class_name or "hand" in class_name:
                    # Check which zone it’s in
                    points_to_check = [
                        (int(x1), int(y1)),
                        (int(x2), int(y1)),
                        (int(x2), int(y2)),
                        (int(x1), int(y2)),
                        (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    ]

                    zone_detected = None
                    for pt in points_to_check:
                        zone = self.get_hand_zone(pt)
                        if zone == "red":
                            zone_detected = "red"
                            break
                        elif zone == "yellow" and zone_detected != "red":
                            zone_detected = "yellow"

                    if zone_detected:
                        hand_detected = True
                        current_zone = zone_detected
                        color = (0, 0, 255) if zone_detected == "red" else (0, 255, 255)

                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(img, (cx, cy), 5, color, -1)

                        current_time = time.time()

                        # 🚨 Save intrusion snapshot if in red zone
                        if zone_detected == "red":
                            if current_time - self.last_intrusion_save_time > self.intrusion_save_cooldown:
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = os.path.join(self.intrusion_save_path, f"intrusion_{timestamp}.jpg")
                                cv2.imwrite(filename, img)
                                print(f"[SAVED] Intrusion image saved: {filename}")
                                self.last_intrusion_save_time = current_time

                        # Alerts (once per second)
                        if current_time - self.last_zone_message_time > self.zone_message_cooldown:
                            if zone_detected != self.current_zone:
                                if zone_detected == "yellow":
                                    print("[WARNING] MACHINE SLOWING DOWN - Glove in Yellow Zone")
                                elif zone_detected == "red":
                                    print("[EMERGENCY] EMERGENCY STOP - Glove in Red Zone")
                                self.current_zone = zone_detected
                                self.last_zone_message_time = current_time

        if not hand_detected and self.current_zone is not None:
            print("[INFO] Glove left safety zone")
            self.current_zone = None

        return hand_detected, current_zone

    def draw_ui(self, img, hand_detected, current_zone):
        """Overlay AOI zones and messages"""
        overlay = img.copy()
        cv2.fillPoly(overlay, [self.compiled_yellow_zone], (0, 255, 255))
        cv2.fillPoly(overlay, [self.compiled_red_zone], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        if self.drawing_mode and len(self.temp_polygon) > 1:
            cv2.polylines(img, [np.array(self.temp_polygon, np.int32)], False, (255, 255, 255), 2)

        if current_zone == "red":
            cv2.putText(img, "STOP", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        elif current_zone == "yellow":
            cv2.putText(img, "WARNING", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
        elif hand_detected:
            cv2.putText(img, "SAFE - GLOVE DETECTED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(img, "INVICTUS SOLUTION", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    def run(self):
        """Main loop"""
        ip_url = "rtsp://admin:Techno%40123@192.168.1.64:554/Streaming/Channels/101"
        cap = cv2.VideoCapture(ip_url)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[ERROR] Failed to open RTSP stream")
            return

        # Start capture thread
        capture_thread = threading.Thread(target=self.frame_capture_thread, args=(cap,), daemon=True)
        capture_thread.start()

        cv2.namedWindow("Hand AOI Monitor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Hand AOI Monitor", self.draw_polygon)

        print("\n[INSTRUCTIONS]")
        print("Press 'D' → Draw AOI polygon (Left-click = add point, Right-click = finalize)")
        print("Press 'R' → Reset polygon to default rectangle")
        print("Press 'ESC' → Exit\n")

        while True:
            try:
                img = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.frame_count += 1

            if self.frame_count % self.frame_skip != 0:
                self.draw_ui(img, False, None)
                cv2.imshow("Hand AOI Monitor", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            hand_detected, current_zone = self.process_yolo_detections(img)
            self.draw_ui(img, hand_detected, current_zone)
            cv2.imshow("Hand AOI Monitor", img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('d'):
                self.drawing_mode = True
                self.temp_polygon = []
                print("[DRAW MODE] Left-click to add points, Right-click to finalize polygon.")
            elif key == ord('r'):
                self.AOI_POLYGON = [(200, 100), (600, 100), (600, 400), (200, 400)]
                self.update_compiled_polygon()
                print("[RESET] AOI Polygon reset to default rectangle.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = OptimizedHandMonitor()
    monitor.run()