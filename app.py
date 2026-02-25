import os
import sys
import cv2
import numpy as np
from datetime import datetime
import threading
import time
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFrame, QSlider, QComboBox,
    QStackedWidget, QTextEdit, QGraphicsDropShadowEffect
)
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QFontDatabase

from engine.vigilus_engine import VigilusEngine
from streamer import Streamer


# --------------------------------
# Styled Status Card (Glassmorphism)
# --------------------------------
class StatusCard(QFrame):
    def __init__(self, title, icon_char=""):
        super().__init__()
        self.setFixedWidth(240)
        self.setStyleSheet("""
            QFrame {
                background: #181818;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 15px;
            }
            QFrame:hover {
                border-color: #00E676;
                background-color: #252525;
            }
        """)
        
        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color:#888; font-size:12px; font-weight:bold; letter-spacing:2px; min-height:15px;")
        self.title_lbl.setAlignment(Qt.AlignLeft)
        
        self.value_lbl = QLabel("NONE")
        self.value_lbl.setStyleSheet("color:white; font-size:22px; font-weight:700; margin-top:5px; min-height:30px;")
        self.value_lbl.setAlignment(Qt.AlignLeft)
        self.value_lbl.setWordWrap(True)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addStretch()

    def update(self, text, color):
        self.value_lbl.setText(text.upper())
        self.value_lbl.setStyleSheet(
            f"font-size:22px; font-weight:700; color:{color}; margin-top:5px; border:none; background:transparent; min-height:30px;"
        )


# --------------------------------
# SETTINGS GROUP (Card-based Layout)
# --------------------------------
class SettingsGroup(QFrame):
    def __init__(self, title, description=""):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color: #00E676; font-size: 13px; font-weight: 800; letter-spacing: 2px; border: none; background: transparent;")
        layout.addWidget(self.title_lbl)

        if description:
            self.desc_lbl = QLabel(description)
            self.desc_lbl.setStyleSheet("color: #666; font-size: 11px; border: none; background: transparent;")
            layout.addWidget(self.desc_lbl)
            
        self.content_lay = QVBoxLayout()
        self.content_lay.setSpacing(10)
        layout.addLayout(self.content_lay)

    def add_widget(self, widget):
        self.content_lay.addWidget(widget)

    def add_layout(self, layout):
        self.content_lay.addLayout(layout)


# --------------------------------
# CAMERA THREAD
# --------------------------------
class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.cap = None

    def run(self):
        # Try DSHOW for Windows first (faster startup, more control)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            if not self.cap.isOpened():
                self.msleep(100)
                continue
                
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.msleep(10)
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()


# --------------------------------
# MAIN APP
# --------------------------------
class VigilusApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VIGILUS — Attention Monitoring System")
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #121212; font-family: 'Segoe UI', sans-serif;")

        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        model_path = os.path.join(base_path, "models/face_landmarker.task")
        self.engine = VigilusEngine(model_path)

        self.cap = None
        self.camera_thread = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        self.sliders = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.active_alert = False  # Track if a popup is currently open
        self.start_time = None
        self.nav_buttons = []
        self.camera_indices = []

        # Initialize Streamer early so UI can access its properties (IP/Port)
        self.streamer = Streamer()
        self.streamer.run()

        self.setMinimumSize(1100, 750)
        self.init_ui()

    # --------------------------------
    # --------------------------------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --------------------------------
        # SIDEBAR
        # --------------------------------
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(260)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #0F0F0F;
                border-right: 1px solid #222;
            }
        """)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(25, 40, 25, 40)
        sidebar_layout.setSpacing(15)

        app_title = QLabel("VIGILUS")
        app_title.setStyleSheet("color:#00E676; font-size:28px; font-weight:900; letter-spacing:5px; margin-bottom: 30px;")
        sidebar_layout.addWidget(app_title)

        nav_links = ["LIVE MONITOR", "SETTINGS", "LOG HISTORY", "ABOUT"]
        for i, name in enumerate(nav_links):
            btn = QPushButton(f"  {name}")
            btn.setCheckable(True)
            if i == 0: btn.setChecked(True)
            self.nav_buttons.append(btn)
            
            self.apply_nav_style(btn, i == 0)
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()
        
        # Session Timer
        self.timer_container = QFrame()
        self.timer_container.setStyleSheet("background: #1A1A1A; border-radius: 10px; padding: 10px; border: 1px solid #222;")
        timer_lay = QVBoxLayout(self.timer_container)
        timer_title = QLabel("SESSION DURATION")
        timer_title.setStyleSheet("color:#555; font-size:9px; font-weight:bold; border:none;")
        self.session_lbl = QLabel("00:00:00")
        self.session_lbl.setStyleSheet("color:#00E676; font-size:18px; font-weight:700; font-family: 'Consolas', monospace; border:none;")
        timer_lay.addWidget(timer_title)
        timer_lay.addWidget(self.session_lbl)
        sidebar_layout.addWidget(self.timer_container)
        
        sidebar_layout.addSpacing(10)

        # Web Status
        self.web_status_lbl = QLabel(f"WEB: ONLINE\n{self.streamer.ip_address}:{self.streamer.port}")
        self.web_status_lbl.setStyleSheet("color:#444; font-size:10px; letter-spacing:1px; line-height:1.5;")
        sidebar_layout.addWidget(self.web_status_lbl)

        main_layout.addWidget(self.sidebar)

        # --------------------------------
        # CONTENT PAGES (Stacked Widget)
        # --------------------------------
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #121212;")
        
        # 1. MONITOR PAGE
        self.monitor_page = QWidget()
        mon_layout = QVBoxLayout(self.monitor_page)
        mon_layout.setContentsMargins(40, 30, 40, 40)
        mon_layout.setSpacing(25)
        
        # Header (Stats & Mode)
        header_bar = QFrame()
        header_bar.setFixedHeight(60)
        header_bar.setStyleSheet("background: #181818; border-radius: 12px; border: 1px solid #252525;")
        header_bar_lay = QHBoxLayout(header_bar)
        header_bar_lay.setContentsMargins(20, 0, 20, 0)
        
        header_title = QLabel("SYSTEM OVERVIEW")
        header_title.setStyleSheet("color: #00E676; font-size: 11px; font-weight: 900; letter-spacing: 2px; border: none;")
        header_bar_lay.addWidget(header_title)
        header_bar_lay.addStretch()
        
        self.fps_lbl = QLabel("FPS: --")
        self.cpu_lbl = QLabel("CPU: 4%")
        self.lat_lbl = QLabel("LAT: --")
        for lbl in [self.fps_lbl, self.cpu_lbl, self.lat_lbl]:
            lbl.setStyleSheet("color: #666; font-size: 10px; font-weight: bold; font-family: 'Consolas', monospace; border: none;")
            header_bar_lay.addWidget(lbl)
            header_bar_lay.addSpacing(15)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Basic", "Intermediate", "Strict"])
        self.mode_box.setCurrentText("Intermediate")
        self.mode_box.setStyleSheet("QComboBox { background: #121212; color: #DDD; border: 1px solid #333; border-radius: 6px; padding-left: 10px; font-size: 11px; }")
        header_bar_lay.addWidget(self.mode_box)
        mon_layout.addWidget(header_bar)

        # Body (Video & Metrics)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(30)

        self.video_wrap = QFrame()
        self.video_wrap.setStyleSheet("QFrame { background-color: #000; border: 2px solid #222; border-radius: 20px; }")
        video_inner = QVBoxLayout(self.video_wrap)
        video_inner.setContentsMargins(5,5,5,5)
        self.video_container = QWidget()
        overlay_layout = QVBoxLayout(self.video_container)
        overlay_layout.setContentsMargins(0,0,0,0)
        self.video = QLabel("STOPPED")
        self.video.setStyleSheet("background-color:transparent; color:#333; font-size:16px; font-weight:bold;")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setFixedSize(720, 405)
        self.hud_top_left = QLabel("CAM_01 // SECURE_STREAM")
        self.hud_bottom_right = QLabel("VIGILUS_CORE_v2.1")
        for hud in [self.hud_top_left, self.hud_bottom_right]:
            hud.setStyleSheet("color: rgba(0, 230, 118, 0.5); font-size: 9px; font-weight: bold; font-family: 'Consolas', monospace; border: none; background: transparent;")
        self.hud_top_left.setParent(self.video)
        self.hud_top_left.move(25, 25)
        self.hud_bottom_right.setParent(self.video)
        self.hud_bottom_right.setFixedWidth(150)
        self.hud_bottom_right.move(560, 365)
        self.live_badge = QLabel(" ● LIVE ")
        self.live_badge.setParent(self.video)
        self.live_badge.move(625, 25)
        self.live_badge.setFixedSize(70, 25)
        self.live_badge.setAlignment(Qt.AlignCenter)
        self.live_badge.setStyleSheet("background-color: rgba(255, 0, 0, 0.8); color: white; font-size: 10px; font-weight: 800; border-radius: 4px;")
        self.live_badge.hide()
        overlay_layout.addWidget(self.video)
        video_inner.addWidget(self.video_container)
        body_layout.addWidget(self.video_wrap)

        metrics_pillar = QVBoxLayout()
        metrics_pillar.setSpacing(20)
        self.card_emotion = StatusCard("EMOTION")
        self.card_gaze = StatusCard("GAZE FOCUS")
        self.card_attention = StatusCard("ATTENTION")
        self.card_cognitive = StatusCard("COGNITIVE")
        metrics_pillar.addWidget(self.card_emotion)
        metrics_pillar.addWidget(self.card_gaze)
        metrics_pillar.addWidget(self.card_attention)
        metrics_pillar.addWidget(self.card_cognitive)

        metrics_pillar.addStretch()
        body_layout.addLayout(metrics_pillar)
        mon_layout.addLayout(body_layout)

        # Footer
        footer_layout = QHBoxLayout()
        self.start_btn = QPushButton("START MONITORING")
        self.start_btn.setFixedSize(220, 55)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.apply_start_btn_style(True)
        self.start_btn.clicked.connect(self.toggle_system)
        footer_layout.addWidget(self.start_btn)
        footer_layout.addStretch()
        mon_layout.addLayout(footer_layout)
        
        self.stack.addWidget(self.monitor_page)

        # 2. SETTINGS PAGE
        self.settings_page = QWidget()
        set_layout = QVBoxLayout(self.settings_page)
        set_layout.setContentsMargins(50, 50, 50, 50)
        set_layout.setSpacing(30)

        st_lbl = QLabel("Application Settings")
        st_lbl.setStyleSheet("color: white; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 10px;")
        set_layout.addWidget(st_lbl)
        
        # Scroll Area for settings
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.scroll_lay = QVBoxLayout(scroll_content)
        self.scroll_lay.setSpacing(25)
        self.scroll_lay.setContentsMargins(0, 0, 20, 0)
        
        # --- GROUP 1: MONITORING PRECISION
        precision_grp = SettingsGroup("Stability & Precision", "Control how sensitive the monitoring engine behaves.")
        
        def add_setting_slider(name, min_v, max_v, default):
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0,0,0,0)
            
            top_lay = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet("color: #AAA; font-size: 11px; font-weight: 600; background: transparent; border: none;")
            val_lbl = QLabel(str(default))
            val_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: 800; background: transparent; border: none;")
            top_lay.addWidget(lbl)
            top_lay.addStretch()
            top_lay.addWidget(val_lbl)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default)
            slider.setStyleSheet("""
                QSlider::groove:horizontal { background: #2A2A2A; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #00E676; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; border: 2px solid #1A1A1A; }
                QSlider::handle:horizontal:hover { background: #69F0AE; }
            """)
            slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
            slider.valueChanged.connect(self.update_engine)
            
            lay.addLayout(top_lay)
            lay.addWidget(slider)
            precision_grp.add_widget(container)
            self.sliders[name] = slider

        add_setting_slider("EYE GAZE SENSITIVITY", 1, 10, 5)
        add_setting_slider("HEAD PITCH SENSITIVITY", 1, 10, 5)
        add_setting_slider("EMOTION STABILITY (ms)", 100, 2000, 800)
        
        self.scroll_lay.addWidget(precision_grp)

        # --- GROUP 2: CAMERA DEVICE
        camera_grp = SettingsGroup("Camera Configuration", "Select and refresh your video input source.")
        
        cam_ui_lay = QHBoxLayout()
        self.camera_box = QComboBox()
        self.camera_box.setFixedSize(220, 42)
        self.camera_box.setStyleSheet("""
            QComboBox { 
                background: #121212; color: #DDD; border: 1px solid #333; 
                border-radius: 8px; padding-left: 15px; font-size: 12px; font-weight: 600;
            }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox:hover { border-color: #00E676; }
        """)
        
        self.refresh_cam_btn = QPushButton(" REFRESH DEVICES")
        self.refresh_cam_btn.setFixedSize(160, 42)
        self.refresh_cam_btn.setStyleSheet("""
            QPushButton {
                background: #252525; color: #FFF; font-size: 11px; font-weight: 800;
                border: 1px solid #333; border-radius: 8px;
            }
            QPushButton:hover { background: #333; border-color: #00E676; color: #00E676; }
        """)
        self.refresh_cam_btn.clicked.connect(self.discover_cameras)
        
        cam_ui_lay.addWidget(self.camera_box)
        cam_ui_lay.addSpacing(15)
        cam_ui_lay.addWidget(self.refresh_cam_btn)
        cam_ui_lay.addStretch()
        
        camera_grp.add_layout(cam_ui_lay)
        self.scroll_lay.addWidget(camera_grp)
        
        # Initial Discovery
        self.discover_cameras()

        self.scroll_lay.addStretch()
        scroll.setWidget(scroll_content)
        set_layout.addWidget(scroll)
        
        self.stack.addWidget(self.settings_page)

        # 3. LOGS PAGE
        self.logs_page = QWidget()
        logs_layout = QVBoxLayout(self.logs_page)
        logs_layout.setContentsMargins(40, 40, 40, 40)
        log_hdr = QHBoxLayout()
        log_hdr_lbl = QLabel("System Analysis Logs")
        log_hdr_lbl.setStyleSheet("color:white; font-size:24px; font-weight:700;")
        log_hdr.addWidget(log_hdr_lbl)
        log_hdr.addStretch()
        ref_btn = QPushButton("REFRESH")
        ref_btn.setFixedSize(100, 35)
        ref_btn.clicked.connect(self.update_log_viewer)
        log_hdr.addWidget(ref_btn)
        logs_layout.addLayout(log_hdr)
        
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background: #0F0F0F; color: #00E676; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid #222; border-radius: 10px; padding: 15px;")
        logs_layout.addWidget(self.log_viewer)
        self.stack.addWidget(self.logs_page)

        # 4. ABOUT PAGE
        self.about_page = QWidget()
        abt_lay = QVBoxLayout(self.about_page)
        abt_lay.setAlignment(Qt.AlignCenter)
        abt_logo = QLabel("VIGILUS")
        abt_logo.setStyleSheet("color:#00E676; font-size:64px; font-weight:900; letter-spacing:12px;")
        abt_lay.addStretch()
        abt_lay.addWidget(abt_logo, 0, Qt.AlignCenter)
        abt_lay.addWidget(QLabel("v2.1.0-STABLE CORE"), 0, Qt.AlignCenter)
        abt_lay.addStretch()
        self.stack.addWidget(self.about_page)

        main_layout.addWidget(self.stack)

    def apply_nav_style(self, btn, active):
        if active:
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left; height: 45px; font-size: 13px; font-weight: 700;
                    letter-spacing: 1px; border: none; background: #1A1A1A;
                    color: white; border-left: 4px solid #00E676; padding-left: 10px;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left; height: 45px; font-size: 13px; font-weight: 600;
                    letter-spacing: 1px; border: none; background: transparent;
                    color: #555; padding-left: 14px;
                }
                QPushButton:hover { color: #00E676; background: #151515; }
            """)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            self.apply_nav_style(btn, i == index)
        if index == 2: self.update_log_viewer()

    def update_log_viewer(self):
        try:
            if os.path.exists("distractions.log"):
                with open("distractions.log", "r") as f:
                    self.log_viewer.setPlainText("".join(f.readlines()[-100:]))
                    self.log_viewer.verticalScrollBar().setValue(self.log_viewer.verticalScrollBar().maximum())
        except Exception: pass

    def apply_start_btn_style(self, starting):
        if starting:
            self.start_btn.setText("START MONITORING")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E676, stop:1 #00C853);
                    color: black;
                    font-size: 14px;
                    font-weight: 800;
                    letter-spacing: 1px;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background: #69F0AE;
                }
            """)
        else:
            self.start_btn.setText("STOP SYSTEM")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: #FF5252;
                    color: white;
                    font-size: 14px;
                    font-weight: 800;
                    letter-spacing: 1px;
                    border-radius: 12px;
                }
                QPushButton:hover {
                    background: #FF8A80;
                }
            """)

    # --------------------------------
    def set_mode(self, mode):
        self.engine.set_mode(mode)
        
        # Update UI sliders to match the new mode presets
        # Block signals to prevent feedback loop
        for s in self.sliders.values():
            s.blockSignals(True)
            
        if "EYE GAZE SENSITIVITY" in self.sliders:
            self.sliders["EYE GAZE SENSITIVITY"].setValue(self.engine.gaze_h_sensitivity)
        
        if "HEAD PITCH SENSITIVITY" in self.sliders:
            self.sliders["HEAD PITCH SENSITIVITY"].setValue(self.engine.pitch_sensitivity)
            
        if "EMOTION STABILITY (ms)" in self.sliders:
            self.sliders["EMOTION STABILITY (ms)"].setValue(int(self.engine.emotion_stability * 1000))
            
        for s in self.sliders.values():
            s.blockSignals(False)

    # --------------------------------
    def discover_cameras(self):
        """Probes camera indices to find available devices."""
        self.camera_box.clear()
        self.camera_indices = []
        
        # Test indices 0-4
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera_indices.append(i)
                self.camera_box.addItem(f"Camera {i}")
                cap.release()
            else:
                # Try without DSHOW just in case
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.camera_indices.append(i)
                    self.camera_box.addItem(f"Camera {i}")
                    cap.release()
        
        if not self.camera_indices:
            self.camera_box.addItem("No Camera Found")
            self.camera_box.setDisabled(True)

    # --------------------------------
    def update_engine(self):
        # Simply map sliders to engine config
        config = {
            "gaze_h_sensitivity": self.sliders["EYE GAZE SENSITIVITY"].value(),
            "pitch_sensitivity": self.sliders["HEAD PITCH SENSITIVITY"].value()
        }
        
        if "EMOTION STABILITY (ms)" in self.sliders:
            config["emotion_stability"] = self.sliders["EMOTION STABILITY (ms)"].value() / 1000
            
        self.engine.update_config(**config)

    # --------------------------------
    def toggle_system(self):
        if not self.is_running:
            # START SYSTEM
            self.apply_start_btn_style(False)
            self.mode_box.setDisabled(True)
            self.start_time = time.time()
            self.live_badge.show()
            if hasattr(self, 'camera_box'): self.camera_box.setDisabled(True)
            if hasattr(self, 'refresh_cam_btn'): self.refresh_cam_btn.setDisabled(True)
            
            # Pulse glow effect
            self.video_wrap.setStyleSheet("background-color:#000; border: 2px solid #00E676; border-radius:20px;")

            # START SYSTEM
            selected_idx = 0
            if self.camera_box.currentIndex() >= 0 and self.camera_indices:
                selected_idx = self.camera_indices[self.camera_box.currentIndex()]

            self.camera_thread = CameraThread(selected_idx)
            self.camera_thread.frame_ready.connect(self.on_frame_received)
            self.camera_thread.start()

            self.timer.start(30)
            self.is_running = True
            
        else:
            # STOP SYSTEM
            self.timer.stop()
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            
            self.video.clear()
            self.video.setText("STOPPED")
            self.live_badge.hide()
            self.video_wrap.setStyleSheet("background-color:#000; border: 2px solid #222; border-radius:20px;")
            self.session_lbl.setText("00:00:00")
            self.start_time = None
            
            self.apply_start_btn_style(True)
            self.mode_box.setDisabled(False)
            if hasattr(self, 'camera_box'): self.camera_box.setDisabled(False)
            if hasattr(self, 'refresh_cam_btn'): self.refresh_cam_btn.setDisabled(False)
            self.is_running = False

    def on_frame_received(self, frame):
        with self.frame_lock:
            self.latest_raw_frame = frame

    # --------------------------------
    def update_frame(self):
        # Session Timer Update
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            self.session_lbl.setText(f"{h:02}:{m:02}:{s:02}")

        frame = None
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                frame = self.latest_raw_frame.copy()
        
        if frame is None:
            return

        frame = cv2.flip(frame, 1)
        
        # Performance: Scale down for engine processing if frame is too big
        # Engine works best around 640x480 or even 480x360
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            engine_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        else:
            engine_frame = frame

        engine_frame = np.ascontiguousarray(engine_frame)

        processed_engine_frame, data = self.engine.process(engine_frame)
        
        # Use processed_engine_frame for display since it has landmarks
        frame = processed_engine_frame
        
        # ---- CHECK ALERTS
        if data.get("face_count", 0) > 1:
            data["emotion"] = "MULTIPLE FACES"
            data["gaze"] = "ALERT"
            data["attention"] = "ALERT"
            data["cognitive"] = "ALERT"
            color = "#FF5252"
        elif not data.get("face_present", True):
            data["emotion"] = "NO FACE"
            data["gaze"] = "NO FACE"
            data["attention"] = "ALERT"
            data["cognitive"] = "ALERT"
            color = "#FF5252"
        else:
            color = None # Use default logic

        # Update Cards
        def get_color(val, default_color):
            if val in ["ALERT", "NO FACE", "MULTIPLE FACES", "DISTRACTED"]: return "#FF5252" # Red
            if val in ["ATTENTIVE", "FOCUSED", "Center"]: return "#00E676" # Green
            if "Happy" in val: return "#00E676"
            return "#FFA726" # Orange/Warn

        self.card_emotion.update(
            data["emotion"].replace("Emotion: ", ""), 
            color if color else get_color(data["emotion"], "#FFA726")
        )
        self.card_gaze.update(
            data["gaze"],
            color if color else get_color(data["gaze"], "#FFA726")
        )
        self.card_attention.update(
            data["attention"],
            color if color else get_color(data["attention"], "#FFA726")
        )
        self.card_cognitive.update(
            data["cognitive"],
            color if color else get_color(data["cognitive"], "#FFA726")
        )

        # ---- UPDATE HEADER STATS
        # Calculate FPS
        now = time.time()
        if not hasattr(self, '_last_time'): self._last_time = now
        dt = now - self._last_time
        self._last_time = now
        fps = int(1/dt) if dt > 0 else 30
        self.fps_lbl.setText(f"FPS: {min(fps, 60)}")
        
        # Mock CPU (for aesthetic)
        self.cpu_lbl.setText(f"CPU: {random.randint(3, 7)}%")
        
        # Latency from data (if available) or mock
        lat = int(data.get('processing_time', 0.015) * 1000)
        self.lat_lbl.setText(f"LAT: {lat}ms")

        # ---- UPDATE STREAMER
        if hasattr(self, 'streamer'):
            self.streamer.update_frame(frame)
            self.streamer.update_metrics(data)

        # ---- VIDEO DISPLAY
        # Resize frame to fit label while maintaining aspect ratio
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(img)
        
        # Scale with KeepAspectRatio to fit the current video label size
        # SmoothTransformation looks better but is slower. 
        # If performance is an issue, use FastTransformation.
        scaled_pixmap = pixmap.scaled(
            self.video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video.setPixmap(scaled_pixmap)

    # --------------------------------
    def closeEvent(self, e):
        if self.timer.isActive():
            self.timer.stop()
        if self.camera_thread:
            self.camera_thread.stop()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VigilusApp()
    window.show()
    sys.exit(app.exec())
