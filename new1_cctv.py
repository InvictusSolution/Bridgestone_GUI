import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime
import os
import json
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageDraw

import subprocess


# Set CustomTkinter Appearance
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class HandDetector:
    def __init__(self):
        self.master_triggered = False

        self.detection_enabled = False
        self.hand_detected = False
        self.detection_count = 0
        self.current_zone = None
        self.last_zone_message_time = 0
        self.zone_message_cooldown = 1.0
        self.last_intrusion_save_time = 0
        self.intrusion_save_cooldown = 1.0

        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_skip = 2
        self.frame_count = 0

        self.reset_zones()

        self.drawing_mode = False
        self.drawing_points = []  # List of collected points for the current zone
        self.current_drawing_zone = None  # Track which zone is being drawn: 'yellow' or 'red'
        self.update_compiled_polygon()

        self.intrusion_save_path = "intrusions"
        os.makedirs(self.intrusion_save_path, exist_ok=True)

        try:
            self.yolo_model = YOLO(r"runs\detect\train\weights\best.pt")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self.yolo_model = None

        self.camera = None
        self.is_capturing = False

    def reset_zones(self):
        # Initialize as empty lists (no default zones)
        self.yellow_zone_points = []
        self.red_zone_points = []
        self.update_compiled_polygon()

    def update_compiled_polygon(self):
        # Yellow zone
        if self.yellow_zone_points:
            self.compiled_yellow_zone = np.array(self.yellow_zone_points, np.int32)
        else:
            self.compiled_yellow_zone = np.empty((0, 2), dtype=np.int32)

        # Red zone
        if self.red_zone_points:
            self.compiled_red_zone = np.array(self.red_zone_points, np.int32)
        else:
            self.compiled_red_zone = np.empty((0, 2), dtype=np.int32)

    def point_in_poly_fast(self, pt, polygon):
        if polygon.size == 0: return False
        return cv2.pointPolygonTest(polygon, tuple(map(int, pt)), False) >= 0

    def get_hand_zone(self, pt):
        if self.point_in_poly_fast(pt, self.compiled_red_zone): return "red"
        if self.point_in_poly_fast(pt, self.compiled_yellow_zone): return "yellow"
        return None

    def start_capture(self, rtsp_url):
        self.camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened(): return False
        self.is_capturing = True
        threading.Thread(target=self.frame_capture_thread, daemon=True).start()
        return True

    def stop_capture(self):
        self.is_capturing = False
        if self.camera: self.camera.release()

    def frame_capture_thread(self):
        while self.is_capturing:
            ret, frame = self.camera.read()
            if not ret: continue
            try:
                if self.frame_queue.full(): self.frame_queue.get_nowait()
                self.frame_queue.put(frame.copy(), block=False)
            except:
                pass

    # def detect_hands(self, frame):
    #     if not self.detection_enabled or self.yolo_model is None:
    #         return frame, False, None
    #
    #     hand_detected, zone = False, None
    #     results = self.yolo_model(frame, verbose=False, conf=0.5)
    #
    #     for r in results:
    #         for box in r.boxes:
    #             cls_id = int(box.cls[0])
    #             name = self.yolo_model.names[cls_id].lower()
    #             if "hand" in name or "glove" in name:
    #                 hand_detected = True
    #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    #                 cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    #                 zone = self.get_hand_zone((cx, cy))
    #
    #                 color = (0, 0, 255) if zone == "red" else (0, 255, 255) if zone == "yellow" else (0, 255, 0)
    #                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    #
    #                 if zone == "red":
    #                     self.detection_count += 1
    #                     timestamp = datetime.now().strftime("%H%M%S")
    #                     cv2.imwrite(f"intrusions/red_{timestamp}.jpg", frame)
    #                 break
    #     return frame, hand_detected, zone

    def detect_hands(self, frame):
        if not self.detection_enabled or self.yolo_model is None:
            return frame, False, None

        hand_detected, zone = False, None
        results = self.yolo_model(frame, verbose=False, conf=0.5)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self.yolo_model.names[cls_id].lower()

                if "hand" in name or "glove" in name:
                    hand_detected = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    zone = self.get_hand_zone((cx, cy))

                    color = (0, 0, 255) if zone == "red" else (0, 255, 255) if zone == "yellow" else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # -------- CALL master.py ONCE --------
                    if zone == "red" and not self.master_triggered:
                        self.master_triggered = False
                        print("Command Sent!")
                        subprocess.Popen(
                            ["python", "master_code.py"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )

                        self.detection_count += 1
                        timestamp = datetime.now().strftime("%H%M%S")
                        cv2.imwrite(f"intrusions/red_{timestamp}.jpg", frame)

                    if zone != "red":
                        self.master_triggered = False
                    # -------------------------------------

                    break

        return frame, hand_detected, zone

    def draw_ui_overlay(self, frame):
        overlay = frame.copy()

        if self.compiled_yellow_zone.size > 0:
            cv2.fillPoly(overlay, [self.compiled_yellow_zone], (0, 255, 255))

        if self.compiled_red_zone.size > 0:
            cv2.fillPoly(overlay, [self.compiled_red_zone], (0, 0, 255))

        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        if self.drawing_mode:
            # Draw points
            draw_color = (0, 255, 255) if self.current_drawing_zone == 'yellow' else (0, 0,
                                                                                      255) if self.current_drawing_zone == 'red' else (
                255, 255, 255)

            # Connect existing points
            if len(self.drawing_points) > 0:
                # Draw circles for each point
                for pt in self.drawing_points:
                    cv2.circle(frame, pt, 5, draw_color, -1)

                # Draw lines connecting them
                if len(self.drawing_points) > 1:
                    pts = np.array(self.drawing_points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, draw_color, 2)

        return frame

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=0.1)
        except:
            return None


class MachineSafetyGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("INVICTUS SOLUTION | Industrial Safety Vision")
        self.geometry("1300x850")

        # Detector and logic
        self.detector = HandDetector()
        self.is_camera_active = False
        self.is_detecting = False
        self.rtsp_url = "rtsp://admin:Techno%40123@192.168.1.2:554/Streaming/Channels/101"

        self.setup_ui()
        self.start_camera()

    # def setup_ui(self):
    #     # 1. Branding Header
    #     self.header = ctk.CTkFrame(self, height=80, corner_radius=0, fg_color="white")
    #     self.header.pack(side="top", fill="x")
    #     self.header.pack_propagate(False)
    #
    #     try:
    #         # Bridgestone Logo - Centered
    #         bs_img = Image.open(r"/Logo/bbstone.png")
    #         # Maintain aspect ratio for resize
    #         bs_ratio = bs_img.width / bs_img.height
    #         bs_new_h = 65
    #         bs_new_w = int(bs_new_h * bs_ratio)
    #         bs_img = bs_img.resize((bs_new_w, bs_new_h), Image.LANCZOS)
    #
    #         self.bs_photo = ctk.CTkImage(light_image=bs_img, dark_image=bs_img, size=(bs_new_w, bs_new_h))
    #         bs_center_label = ctk.CTkLabel(self.header, image=self.bs_photo, text="")
    #         bs_center_label.place(relx=0.5, rely=0.5, anchor="center")
    #
    #         # Invictus Logo - Right Corner
    #         # inv_img = Image.open("Logo/Invictus_Light.png")
    #         # inv_ratio = inv_img.width / inv_img.height
    #         # inv_new_h = 60
    #         # inv_new_w = int(inv_new_h * inv_ratio)
    #         # inv_img = inv_img.resize((inv_new_w, inv_new_h), Image.LANCZOS)
    #         # self.inv_photo = ctk.CTkImage(light_image=inv_img, dark_image=inv_img, size=(inv_new_w, inv_new_h))
    #         # inv_right_label = ctk.CTkLabel(self.header, image=self.inv_photo, text="")
    #         # inv_right_label.pack(side="right", padx=20)
    #
    #     except Exception as e:
    #         ctk.CTkLabel(self.header, text=f"LOGO ERROR: {e}", text_color="red").pack(side="left", padx=10)
    #
    #     # 2. Main Content Area
    #     self.main_container = ctk.CTkFrame(self)
    #     self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
    #
    #     # Left Column: Video & Logs
    #     self.left_col = ctk.CTkFrame(self.main_container, fg_color="transparent")
    #     self.left_col.pack(side="left", fill="both", expand=True)
    #
    #     # Live Monitoring Frame
    #     self.cam_frame = ctk.CTkFrame(self.left_col)
    #     self.cam_frame.pack(fill="both", expand=True, pady=(0, 10))
    #     self.cam_frame.pack_propagate(False)  # 🔴 CRITICAL
    #
    #     self.cam_header = ctk.CTkLabel(self.cam_frame, text="LIVE MONITORING", font=("Arial", 14, "bold"))
    #     self.cam_header.pack(pady=5)
    #
    #     self.camera_label = tk.Label(self.cam_frame, bg="black")
    #     self.camera_label.pack(fill="both", expand=True, padx=5, pady=5)
    #
    #     self.camera_label.update_idletasks()
    #     self.camera_label.config(width=800, height=450)  # stable aspect
    #
    #     # Logs Frame
    #     self.log_frame = ctk.CTkFrame(self.left_col, height=150)
    #     self.log_frame.pack(fill="x")
    #     self.log_frame.pack_propagate(False)
    #
    #     self.log_header = ctk.CTkLabel(self.log_frame, text="SYSTEM EVENTS", font=("Arial", 12, "bold"))
    #     self.log_header.pack(pady=2)
    #
    #     self.log_text = scrolledtext.ScrolledText(self.log_frame, height=6, bg="#1e1e1e", fg="#00ff00",
    #                                               font=("Consolas", 10), relief="flat")
    #     self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
    #
    #     # Right Column: Controls & Status
    #     self.right_col = ctk.CTkFrame(self.main_container, width=300)
    #     self.right_col.pack(side="right", fill="y", padx=(10, 0))
    #     self.right_col.pack_propagate(False)
    #
    #     # Status Panel
    #     self.status_panel = ctk.CTkFrame(self.right_col)
    #     self.status_panel.pack(fill="x", pady=(0, 10), padx=5)
    #
    #     ctk.CTkLabel(self.status_panel, text="ZONE STATUS", font=("Arial", 14, "bold")).pack(pady=10)
    #
    #     self.zone_indicator = ctk.CTkCanvas(self.status_panel, width=100, height=100, bg="#2b2b2b",
    #                                         highlightthickness=0)
    #     self.zone_indicator.pack(pady=5)
    #     # Match canvas background to theme manually if needed, or keep it neutral
    #     self.indicator_circle = self.zone_indicator.create_oval(10, 10, 90, 90, fill="#cccccc")
    #
    #     self.lbl_hand = ctk.CTkLabel(self.status_panel, text="HAND: NOT DETECTED", font=("Arial", 14, "bold"),
    #                                  text_color="white")
    #     self.lbl_hand.pack(pady=5)
    #
    #     self.lbl_count = ctk.CTkLabel(self.status_panel, text="INTRUSIONS: 0", font=("Arial", 12))
    #     self.lbl_count.pack(pady=(0, 15))
    #
    #     # Controls Panel
    #     self.ctrl_panel = ctk.CTkFrame(self.right_col)
    #     self.ctrl_panel.pack(fill="both", expand=True, padx=5, pady=5)
    #
    #     ctk.CTkLabel(self.ctrl_panel, text="COMMANDS", font=("Arial", 14, "bold")).pack(pady=10)
    #
    #     self.btn_draw_yellow = ctk.CTkButton(self.ctrl_panel, text="🟡 DRAW YELLOW ZONE",
    #                                          command=self.enable_yellow_zone_draw, height=40, fg_color="#FFD700",
    #                                          hover_color="#FFA500", text_color="black", font=("Arial", 12, "bold"))
    #     self.btn_draw_yellow.pack(fill="x", padx=10, pady=5)
    #
    #     self.btn_draw_red = ctk.CTkButton(self.ctrl_panel, text="🔴 DRAW RED ZONE", command=self.enable_red_zone_draw,
    #                                       height=40, fg_color="#d32f2f", hover_color="#b71c1c",
    #                                       font=("Arial", 12, "bold"))
    #     self.btn_draw_red.pack(fill="x", padx=10, pady=5)
    #
    #     self.btn_reset = ctk.CTkButton(self.ctrl_panel, text="🔄 RESET AOI", command=self.reset_aoi, height=40,
    #                                    fg_color="gray", font=("Arial", 12, "bold"))
    #     self.btn_reset.pack(fill="x", padx=10, pady=5)
    #
    #     self.btn_clear = ctk.CTkButton(self.ctrl_panel, text="🧹 CLEAR LOGS",
    #                                    command=lambda: self.log_text.delete(1.0, tk.END), height=40, fg_color="gray",
    #                                    font=("Arial", 12, "bold"))
    #     self.btn_clear.pack(fill="x", padx=10, pady=5)
    #
    #     self.btn_emergency = ctk.CTkButton(self.ctrl_panel, text="🛑 EMERGENCY STOP", command=self.emergency_stop,
    #                                        height=60, fg_color="#d32f2f", hover_color="#b71c1c",
    #                                        font=("Arial", 14, "bold"))
    #     self.btn_emergency.pack(side="bottom", fill="x", padx=10, pady=20)

    def setup_ui(self):
        """
        Setup the main UI layout including:
        - Branding header with logos
        - Live monitoring video feed
        - System events log
        - Control panel with zone status
        """

        # ============================================================================
        # 1. BRANDING HEADER - Fixed position at the top
        # ============================================================================
        self.header = ctk.CTkFrame(
            self,
            height=80,
            corner_radius=0,
            fg_color="white"
        )
        self.header.pack(side="top", fill="x")
        self.header.pack_propagate(False)  # Maintain fixed height

        # ============================================================================
        # LOGO HANDLING
        # ============================================================================
        try:
            # Load and resize Bridgestone logo

            bs_img = Image.open("Logo/bbstone.png")

            # Calculate dimensions while maintaining aspect ratio
            bs_ratio = bs_img.width / bs_img.height
            bs_new_h = 65
            bs_new_w = int(bs_new_h * bs_ratio)

            bs_img = bs_img.resize((bs_new_w, bs_new_h), Image.LANCZOS)

            # Create CTkImage for the logo
            self.bs_photo = ctk.CTkImage(
                light_image=bs_img,
                dark_image=bs_img,
                size=(bs_new_w, bs_new_h)
            )

            # Position logo at the center of header
            bs_center_label = ctk.CTkLabel(
                self.header,
                image=self.bs_photo,
                text=""
            )
            bs_center_label.place(relx=0.5, rely=0.5, anchor="center")

            """
            # Uncomment to add Invictus logo on the right
            inv_img = Image.open("Logo/Invictus_Light.png")
            inv_ratio = inv_img.width / inv_img.height
            inv_new_h = 60
            inv_new_w = int(inv_new_h * inv_ratio)
            inv_img = inv_img.resize((inv_new_w, inv_new_h), Image.LANCZOS)

            self.inv_photo = ctk.CTkImage(
                light_image=inv_img, 
                dark_image=inv_img, 
                size=(inv_new_w, inv_new_h)
            )

            inv_right_label = ctk.CTkLabel(
                self.header, 
                image=self.inv_photo, 
                text=""
            )
            inv_right_label.pack(side="right", padx=20)
            """

        except FileNotFoundError:
            # If logo file is missing, show error message
            error_msg = ctk.CTkLabel(
                self.header,
                text="⚠️ LOGO FILE NOT FOUND: /Logo/bbstone.png",
                text_color="red",
                font=("Arial", 12)
            )
            error_msg.pack(expand=True)
            print("ERROR: Logo file not found at /Logo/bbstone.png")

        except Exception as e:
            # Generic error handling for other issues
            error_msg = ctk.CTkLabel(
                self.header,
                text=f"⚠️ LOGO ERROR: {str(e)[:50]}...",
                text_color="red",
                font=("Arial", 12)
            )
            error_msg.pack(expand=True)
            print(f"Logo loading error: {e}")

        # ============================================================================
        # 2. MAIN CONTENT AREA
        # ============================================================================
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ============================================================================
        # LEFT COLUMN: Video Feed & Logs
        # ============================================================================
        self.left_col = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        self.left_col.pack(side="left", fill="both", expand=True)

        # ----------------------------------------------------------------------------
        # 2.1 LIVE MONITORING FRAME
        # ----------------------------------------------------------------------------
        self.cam_frame = ctk.CTkFrame(self.left_col)
        self.cam_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.cam_frame.pack_propagate(False)  # Maintain fixed layout

        # Camera header
        self.cam_header = ctk.CTkLabel(
            self.cam_frame,
            text="LIVE MONITORING",
            font=("Arial", 14, "bold")
        )
        self.cam_header.pack(pady=5)

        # Camera display area
        self.camera_label = tk.Label(
            self.cam_frame,
            bg="black",
            relief="flat"
        )
        self.camera_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Set fixed dimensions for video feed
        self.camera_label.update_idletasks()
        self.camera_label.config(width=800, height=450)  # 16:9 aspect ratio

        # ----------------------------------------------------------------------------
        # 2.2 SYSTEM LOGS FRAME
        # ----------------------------------------------------------------------------
        self.log_frame = ctk.CTkFrame(
            self.left_col,
            height=150
        )
        self.log_frame.pack(fill="x")
        self.log_frame.pack_propagate(False)  # Maintain fixed height

        # Logs header
        self.log_header = ctk.CTkLabel(
            self.log_frame,
            text="SYSTEM EVENTS",
            font=("Arial", 12, "bold")
        )
        self.log_header.pack(pady=2)

        # Scrollable text area for logs
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame,
            height=6,
            bg="#1e1e1e",  # Dark background
            fg="#00ff00",  # Green text (matrix style)
            font=("Consolas", 10),  # Monospaced font
            relief="flat",
            insertbackground="#00ff00"  # Green cursor
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # ============================================================================
        # RIGHT COLUMN: Controls & Status Panel
        # ============================================================================
        self.right_col = ctk.CTkFrame(
            self.main_container,
            width=300
        )
        self.right_col.pack(side="right", fill="y", padx=(10, 0))
        self.right_col.pack_propagate(False)

        # ----------------------------------------------------------------------------
        # 3.1 STATUS PANEL
        # ----------------------------------------------------------------------------
        self.status_panel = ctk.CTkFrame(self.right_col)
        self.status_panel.pack(fill="x", pady=(0, 10), padx=5)

        # Zone status header
        ctk.CTkLabel(
            self.status_panel,
            text="ZONE STATUS",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Visual indicator (circle canvas)
        self.zone_indicator = ctk.CTkCanvas(
            self.status_panel,
            width=100,
            height=100,
            bg="#2b2b2b",
            highlightthickness=0
        )
        self.zone_indicator.pack(pady=5)

        # Create the status circle (default: gray)
        self.indicator_circle = self.zone_indicator.create_oval(
            10, 10, 90, 90,
            fill="#cccccc",  # Default gray
            outline="#666666",  # Border
            width=2
        )

        # Hand detection status
        self.lbl_hand = ctk.CTkLabel(
            self.status_panel,
            text="HAND: NOT DETECTED",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.lbl_hand.pack(pady=5)

        # Intrusion counter
        self.lbl_count = ctk.CTkLabel(
            self.status_panel,
            text="INTRUSIONS: 0",
            font=("Arial", 12),
            text_color="#90CAF9"  # Light blue
        )
        self.lbl_count.pack(pady=(0, 15))

        # ----------------------------------------------------------------------------
        # 3.2 CONTROLS PANEL
        # ----------------------------------------------------------------------------
        self.ctrl_panel = ctk.CTkFrame(self.right_col)
        self.ctrl_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # Controls header
        ctk.CTkLabel(
            self.ctrl_panel,
            text="COMMANDS",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Button: Draw Yellow Zone
        self.btn_draw_yellow = ctk.CTkButton(
            self.ctrl_panel,
            text="🟡 DRAW YELLOW ZONE",
            command=self.enable_yellow_zone_draw,
            height=40,
            fg_color="#FFD700",  # Gold
            hover_color="#FFA500",  # Darker orange
            text_color="black",
            font=("Arial", 12, "bold")
        )
        self.btn_draw_yellow.pack(fill="x", padx=10, pady=5)

        # Button: Draw Red Zone
        self.btn_draw_red = ctk.CTkButton(
            self.ctrl_panel,
            text="🔴 DRAW RED ZONE",
            command=self.enable_red_zone_draw,
            height=40,
            fg_color="#d32f2f",  # Red
            hover_color="#b71c1c",  # Darker red
            font=("Arial", 12, "bold")
        )
        self.btn_draw_red.pack(fill="x", padx=10, pady=5)

        # Button: Reset AOI
        self.btn_reset = ctk.CTkButton(
            self.ctrl_panel,
            text="🔄 RESET AOI",
            command=self.reset_aoi,
            height=40,
            fg_color="gray",
            hover_color="#5d5d5d",
            font=("Arial", 12, "bold")
        )
        self.btn_reset.pack(fill="x", padx=10, pady=5)

        # Button: Clear Logs
        self.btn_clear = ctk.CTkButton(
            self.ctrl_panel,
            text="🧹 CLEAR LOGS",
            command=lambda: self.log_text.delete(1.0, tk.END),
            height=40,
            fg_color="gray",
            hover_color="#5d5d5d",
            font=("Arial", 12, "bold")
        )
        self.btn_clear.pack(fill="x", padx=10, pady=5)

        # Button: Emergency Stop (prominent)
        self.btn_emergency = ctk.CTkButton(
            self.ctrl_panel,
            text="🛑 EMERGENCY STOP",
            command=self.emergency_stop,
            height=60,
            fg_color="#d32f2f",  # Red
            hover_color="#b71c1c",  # Darker red
            font=("Arial", 14, "bold"),
            corner_radius=10
        )
        self.btn_emergency.pack(side="bottom", fill="x", padx=10, pady=20)

    def log_message(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def start_camera(self):
        if self.detector.start_capture(self.rtsp_url):
            # if self.detector.start_capture():
            self.is_camera_active = True
            self.update_feed()
            self.log_message("Stream initialized.")
        else:
            self.log_message("Failed to connect to camera.")

    def update_feed(self):
        frame = self.detector.get_frame()
        if frame is not None:
            self.last_frame = frame.copy()

            if self.is_detecting:
                frame, detected, zone = self.detector.detect_hands(frame)
                self.update_status_ui(detected, zone)

            frame = self.detector.draw_ui_overlay(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            w_target = self.camera_label.winfo_width()
            h_target = self.camera_label.winfo_height()

            if w_target > 10 and h_target > 10:
                h, w = frame.shape[:2]

                # 🔴 KEEP ASPECT RATIO
                scale = min(w_target / w, h_target / h)

                new_w = int(w * scale)
                new_h = int(h * scale)

                resized = cv2.resize(frame, (new_w, new_h))

                # 🔴 CREATE BLACK CANVAS
                canvas = np.zeros((h_target, w_target, 3), dtype=np.uint8)

                x_offset = (w_target - new_w) // 2
                y_offset = (h_target - new_h) // 2

                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

                # 🔴 SAVE TRANSFORM FOR MOUSE → FRAME MAPPING
                self.display_scale = scale
                self.display_x_offset = x_offset
                self.display_y_offset = y_offset

                frame = canvas

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.camera_label.configure(image=img)
            self.camera_label.image = img

        self.after(10, self.update_feed)

    def update_status_ui(self, detected, zone):
        color = "#cccccc"
        text_color = "white"  # Default for dark mode

        if zone == "red":
            color = "#ff0000"
        elif zone == "yellow":
            color = "#ffff00"
        elif detected:
            color = "#00ff00"

        self.zone_indicator.itemconfig(self.indicator_circle, fill=color)

        status_text = "DETECTED" if detected else "NOT DETECTED"
        status_fg = "#ff4444" if detected else "white"  # distinctive color for text

        self.lbl_hand.configure(text=f"HAND: {status_text}", text_color=status_fg)
        self.lbl_count.configure(text=f"INTRUSIONS: {self.detector.detection_count}")

    def reset_aoi(self):
        self.detector.reset_zones()
        self.log_message("Zones reset to default.")

    def enable_yellow_zone_draw(self):
        self.detector.drawing_mode = True
        self.detector.current_drawing_zone = 'yellow'
        self.detector.drawing_points = []
        self.camera_label.bind('<ButtonRelease-1>', self.handle_canvas_click)
        self.log_message("Yellow Zone Draw Mode: Click 4 points on video.")

    def enable_red_zone_draw(self):
        self.detector.drawing_mode = True
        self.detector.current_drawing_zone = 'red'
        self.detector.drawing_points = []
        self.camera_label.bind('<ButtonRelease-1>', self.handle_canvas_click)
        self.log_message("Red Zone Draw Mode: Click 4 points on video.")

    def handle_canvas_click(self, e):
        if not hasattr(self, "display_scale"): return

        x = int((e.x - self.display_x_offset) / self.display_scale)
        y = int((e.y - self.display_y_offset) / self.display_scale)

        self.detector.drawing_points.append((x, y))
        self.log_message(f"Point {len(self.detector.drawing_points)} set at ({x}, {y})")

        if len(self.detector.drawing_points) == 4:
            if self.detector.current_drawing_zone == 'yellow':
                self.detector.yellow_zone_points = self.detector.drawing_points.copy()
                self.log_message("Yellow Zone Set. Detection Active.")
            elif self.detector.current_drawing_zone == 'red':
                self.detector.red_zone_points = self.detector.drawing_points.copy()
                self.log_message("Red Zone Set. Detection Active.")

            self.detector.update_compiled_polygon()
            self.detector.drawing_mode = False
            self.detector.current_drawing_zone = None
            self.is_detecting = True
            self.detector.detection_enabled = True

            self.camera_label.unbind('<ButtonRelease-1>')

    def emergency_stop(self):
        self.is_detecting = False
        self.log_message("!!! EMERGENCY STOP TRIGGERED !!!")
        messagebox.showwarning("Emergency", "Machine Stop Signal Sent!")


if __name__ == "__main__":
    app = MachineSafetyGUI()
    app.mainloop()