from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

import cv2


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()

        self.engine = engine

        self.setWindowTitle("Attention Monitoring System")
        self.setMinimumSize(1280, 720)

        # -----------------------------
        # Central widget
        # -----------------------------
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # -----------------------------
        # LEFT: Video Feed
        # -----------------------------
        self.video_label = QLabel("Camera feed loading...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: black; color: white; font-size: 18px;"
        )

        main_layout.addWidget(self.video_label, stretch=3)

        # -----------------------------
        # RIGHT: Info Panel
        # -----------------------------
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setAlignment(Qt.AlignTop)

        self.status_label = self._make_label("Status: ---")
        self.emotion_label = self._make_label("Emotion: ---")
        self.gaze_label = self._make_label("Gaze: ---")
        self.attention_label = self._make_label("Attention: ---")
        self.attention_time_label = self._make_label("Attention Time: ---")
        self.cognitive_label = self._make_label("Cognitive: ---")
        self.cognitive_time_label = self._make_label("Cognitive Time: ---")

        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.emotion_label)
        info_layout.addWidget(self.gaze_label)
        info_layout.addWidget(self.attention_label)
        info_layout.addWidget(self.attention_time_label)
        info_layout.addWidget(self.cognitive_label)
        info_layout.addWidget(self.cognitive_time_label)

        main_layout.addWidget(info_panel, stretch=1)

        # -----------------------------
        # UI Update Timer (30 FPS)
        # -----------------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

    # --------------------------------------------------
    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 16px;
                padding: 6px;
            }
            """
        )
        return lbl

    # --------------------------------------------------
    def update_ui(self):
        # -------- Frame
        frame = self.engine.get_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            qimg = QImage(
                rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        # -------- State
        state = self.engine.get_state()

        self.status_label.setText(f"Status: {state['status']}")
        self.emotion_label.setText(f"Emotion: {state['emotion']}")
        self.gaze_label.setText(f"Gaze: {state['gaze']}")
        self.attention_label.setText(f"Attention: {state['attention']}")
        self.attention_time_label.setText(
            f"Attention Time: {state['attention_duration']:.1f} s"
        )
        self.cognitive_label.setText(f"Cognitive: {state['cognitive']}")
        self.cognitive_time_label.setText(
            f"Cognitive Time: {state['cognitive_duration']:.1f} s"
        )

    # --------------------------------------------------
    def closeEvent(self, event):
        self.engine.stop()
        event.accept()
