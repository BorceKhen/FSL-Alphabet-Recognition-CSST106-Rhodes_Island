"""Real-time PyQt6 + OpenCV GUI for FSL Alphabet Recognition & Word Formation."""

from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONFIDENCE_THRESHOLD,
    TARGET_FPS,
)
from src.features import HandLandmarkExtractor, draw_hand_skeleton
from src.postprocess import WordFormationProcessor
from src.predictor import ModelPredictor


class FSLMainWindow(QMainWindow):
    """Main desktop window for real-time FSL fingerspelling transcription."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FSL Alphabet Recognition & Word Formation System")
        self.setMinimumSize(1080, 680)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #12151c; color: #e8ecf2; }
            QLabel { color: #e8ecf2; }
            QFrame#Card {
                background-color: #1b202b;
                border: 1px solid #2b3242;
                border-radius: 10px;
            }
            QPushButton {
                background-color: #273042;
                color: #e8ecf2;
                border: 1px solid #39445c;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #344058; }
            QPushButton#Danger { background-color: #5c2832; border-color: #7d3744; }
            QPushButton#Danger:hover { background-color: #743340; }
            QPushButton#Accent { background-color: #1f5c4a; border-color: #2a7a63; }
            QPushButton#Accent:hover { background-color: #27735d; }
            QTextEdit {
                background-color: #0e1117;
                color: #f4f7fb;
                border: 1px solid #2b3242;
                border-radius: 8px;
                padding: 10px;
                font-size: 24px;
                font-weight: 600;
            }
            QProgressBar {
                background-color: #0e1117;
                border: 1px solid #2b3242;
                border-radius: 5px;
                text-align: center;
                color: #e8ecf2;
                height: 20px;
            }
            QProgressBar::chunk { background-color: #2ecc71; border-radius: 4px; }
            """
        )

        self.predictor = ModelPredictor()
        self.processor = WordFormationProcessor()
        self.extractor = HandLandmarkExtractor(running_mode="VIDEO")
        self._start_time = time.monotonic()

        self.cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(int(1000 / TARGET_FPS))

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left column: Live webcam feed + status
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        self.video_label = QLabel("Initializing camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #0b0d12; border-radius: 6px;")
        left_layout.addWidget(self.video_label)

        controls_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItem("Deep MLP")
        if self.predictor.svm is not None:
            self.model_combo.addItem("Baseline SVM")
        self.model_combo.setStyleSheet(
            "background-color: #273042; color: #e8ecf2; padding: 6px; border-radius: 6px;"
        )

        self.clahe_checkbox = QCheckBox("Adaptive Lighting (CLAHE)")
        self.clahe_checkbox.setStyleSheet("color: #cfd6e4; font-size: 13px;")

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #f39c12; font-size: 13px; font-weight: 600;")

        controls_row.addWidget(QLabel("Model:"))
        controls_row.addWidget(self.model_combo)
        controls_row.addSpacing(12)
        controls_row.addWidget(self.clahe_checkbox)
        controls_row.addStretch()
        controls_row.addWidget(self.status_label)
        left_layout.addLayout(controls_row)

        # Right column: Prediction telemetry + Transcription box
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(14)

        pred_title = QLabel("DETECTED SIGN")
        pred_title.setStyleSheet("color: #8b95a8; font-size: 12px; font-weight: 700;")
        right_layout.addWidget(pred_title)

        self.letter_label = QLabel("-")
        self.letter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.letter_label.setFont(QFont("Segoe UI", 64, QFont.Weight.Bold))
        self.letter_label.setStyleSheet(
            "background-color: #0e1117; border: 1px solid #2b3242; border-radius: 10px; padding: 12px;"
        )
        right_layout.addWidget(self.letter_label)

        conf_lbl = QLabel(f"Confidence (Threshold: {int(CONFIDENCE_THRESHOLD * 100)}%)")
        conf_lbl.setStyleSheet("color: #b4bece; font-size: 12px;")
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        right_layout.addWidget(conf_lbl)
        right_layout.addWidget(self.conf_bar)

        dwell_lbl = QLabel("Hold Confirmation (1.0s Dwell)")
        dwell_lbl.setStyleSheet("color: #b4bece; font-size: 12px;")
        self.dwell_bar = QProgressBar()
        self.dwell_bar.setRange(0, 100)
        self.dwell_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #3498db; border-radius: 4px; }"
        )
        right_layout.addWidget(dwell_lbl)
        right_layout.addWidget(self.dwell_bar)

        text_title = QLabel("TRANSCRIPTION BUFFER")
        text_title.setStyleSheet("color: #8b95a8; font-size: 12px; font-weight: 700;")
        right_layout.addWidget(text_title)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlaceholderText("Hold a sign for 1.0s to type...")
        right_layout.addWidget(self.text_box, stretch=1)

        btn_row = QHBoxLayout()
        space_btn = QPushButton("Space")
        space_btn.setObjectName("Accent")
        space_btn.clicked.connect(self._on_space)

        back_btn = QPushButton("Backspace")
        back_btn.clicked.connect(self._on_backspace)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("Danger")
        clear_btn.clicked.connect(self._on_clear)

        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._on_copy)

        btn_row.addWidget(space_btn)
        btn_row.addWidget(back_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(copy_btn)
        right_layout.addLayout(btn_row)

        root.addWidget(left_card, stretch=3)
        root.addWidget(right_card, stretch=2)

    def _on_space(self) -> None:
        self.text_box.setPlainText(self.processor.insert_space())

    def _on_backspace(self) -> None:
        self.text_box.setPlainText(self.processor.backspace())

    def _on_clear(self) -> None:
        self.text_box.setPlainText(self.processor.clear())

    def _on_copy(self) -> None:
        QApplication.clipboard().setText(self.processor.get_text())
        self.status_label.setText("Copied to clipboard")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._on_space()
        elif event.key() == Qt.Key.Key_Backspace:
            self._on_backspace()
        elif event.key() == Qt.Key.Key_Escape:
            self._on_clear()
        else:
            super().keyPressEvent(event)

    def _on_frame(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self.video_label.setText("Webcam not available. Connect a camera and restart.")
            self.status_label.setText("No Camera")
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText("Camera read error")
            return

        frame = cv2.flip(frame, 1)
        now_sec = time.monotonic() - self._start_time
        ts_ms = int(now_sec * 1000)

        vec, landmarks = self.extractor.extract(
            frame,
            timestamp_ms=ts_ms,
            use_clahe=self.clahe_checkbox.isChecked(),
        )

        raw_label: Optional[str] = None
        raw_conf = 0.0
        if vec is not None and landmarks is not None:
            frame = draw_hand_skeleton(frame, landmarks)
            raw_label, raw_conf = self.predictor.predict(vec, self.model_combo.currentText())
            self.status_label.setText("Hand Tracked")
            self.status_label.setStyleSheet("color: #2ecc71; font-size: 13px; font-weight: 600;")
        else:
            self.status_label.setText("No hand detected — adjust position or lighting")
            self.status_label.setStyleSheet("color: #f39c12; font-size: 13px; font-weight: 600;")

        state = self.processor.update(raw_label, raw_conf, now_sec)

        self.letter_label.setText(state.smoothed_label or (state.raw_label or "-"))
        self.conf_bar.setValue(int(state.raw_confidence * 100))
        self.dwell_bar.setValue(int(state.dwell_progress * 100))

        if state.just_confirmed:
            self.text_box.setPlainText(state.text_buffer)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.extractor.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = FSLMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
