import sys
import os
import cv2
import numpy as np
import time
import onnxruntime as ort
from collections import Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFrame, QPushButton, QFileDialog,
                             QProgressBar, QStackedWidget, QGridLayout, QHeaderView, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from model_utils import ensure_optimized_model

# Human-readable GTSRB Traffic Sign Mapping
GTSRB_CLASSES = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No Passing", 10: "No Passing Heavy Vech", 11: "Right-of-Way",
    12: "Priority Road", 13: "Yield", 14: "STOP", 15: "No Vehicles", 16: "Heavy Vech Prohibited",
    17: "No Entry", 18: "General Caution", 19: "Dangerous Curve Left", 20: "Dangerous Curve Right",
    21: "Double Curve", 22: "Bumpy Road", 23: "Slippery Road", 24: "Road Narrows Right",
    25: "Road Work", 26: "Traffic Signals", 27: "Pedestrians", 28: "Children Crossing",
    29: "Bicycles Crossing", 30: "Ice/Snow Caution", 31: "Wild Animals", 32: "End Limits",
    33: "Turn Right Ahead", 34: "Turn Left Ahead", 35: "Ahead Only", 36: "Go Straight or Right",
    37: "Go Straight or Left", 38: "Keep Right", 39: "Keep Left", 40: "Roundabout Mandatory",
    41: "End No Passing", 42: "End No Passing Heavy Vech"
}

def safe_softmax(x):
    """Robust 1D Softmax calculation regardless of input dimensions."""
    arr = np.array(x, dtype=np.float32).flatten()
    e_x = np.exp(arr - np.max(arr))
    return e_x / e_x.sum()

# ---------------------------------------------------------
# Thread for low-latency NPU Inference & Pipeline Tracker
# ---------------------------------------------------------
class InferenceThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray) # frame, roi_crop
    update_pipeline_signal = pyqtSignal(dict, list)          # telemetry, top3 predictions
    session_ended_signal = pyqtSignal(dict)                  # final session summary statistics

    def __init__(self):
        super().__init__()
        self.running = True
        self.video_source = 0  
        self.source_changed = False
        
        self.reset_stats()
        
        # Absolute Path Resolution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = ensure_optimized_model()
        
        print(f"Loading ONNX Model from: {model_path}")
        session_options = ort.SessionOptions()
        
        self.provider_name = "CPU Fallback"
        if model_path and os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=['QNNExecutionProvider', 'CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.model_loaded = True
                self.provider_name = self.session.get_providers()[0]
                print("[SUCCESS] Loaded model via QNN/ONNX Provider!")
            except Exception as e:
                print(f"[WARNING] QNN Failed ({e}), falling back to CPU Execution Provider...")
                try:
                    self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=['CPUExecutionProvider'])
                    self.input_name = self.session.get_inputs()[0].name
                    self.model_loaded = True
                    print("[SUCCESS] Loaded model via CPU Provider!")
                except Exception as e_cpu:
                    print(f"[ERROR] Model loading failed completely: {e_cpu}")
                    self.model_loaded = False
        else:
            print("[WARNING] Model file not found. Running in Demo Mode.")
            self.model_loaded = False

    def reset_stats(self):
        self.total_frames = 0
        self.total_latency = 0.0
        self.detected_classes = []
        self.start_timestamp = time.time()

    def set_source(self, source):
        self.video_source = source
        self.source_changed = True

    def run(self):
        self.reset_stats()
        cap = cv2.VideoCapture(self.video_source)
        
        while self.running:
            if self.source_changed:
                cap.release()
                self.reset_stats()
                cap = cv2.VideoCapture(self.video_source)
                self.source_changed = False

            if not cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            
            # Feed ended
            if not ret:
                cap.release()
                self.trigger_session_end()
                break
            
            t0 = time.time()
            
            # Crop ROI Center 32x32
            h, w, _ = frame.shape
            crop_size = min(h, w) // 2
            cy, cx = h // 2, w // 2
            roi_raw = frame[max(0, cy-crop_size//2):min(h, cy+crop_size//2), 
                            max(0, cx-crop_size//2):min(w, cx+crop_size//2)]
            
            if roi_raw.size == 0:
                roi_raw = frame
                
            roi_32 = cv2.resize(roi_raw, (32, 32))
            img_normalized = np.expand_dims(roi_32.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
            img_normalized = (img_normalized - 0.5) / 0.5
            
            # Draw Bounding Box
            cv2.rectangle(frame, (cx-crop_size//2, cy-crop_size//2), (cx+crop_size//2, cy+crop_size//2), (0, 229, 255), 2)
            
            top3 = []
            if self.model_loaded:
                try:
                    inputs = {self.input_name: img_normalized}
                    raw_out = self.session.run(None, inputs)[0]
                    
                    probs = safe_softmax(raw_out)
                    top_indices = np.argsort(probs)[-3:][::-1]
                    
                    for idx in top_indices:
                        label = GTSRB_CLASSES.get(idx, f"Class {idx}")
                        top3.append((label, float(probs[idx])))
                    
                    detected_class_id = top_indices[0]
                    self.detected_classes.append(GTSRB_CLASSES.get(detected_class_id, f"Class {detected_class_id}"))
                    
                    # Overlay top prediction
                    cv2.putText(frame, top3[0][0], (cx-crop_size//2, max(30, cy-crop_size//2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as err:
                    print(f"[INFERENCE ERROR] {err}")
                    top3 = [("Inference Error", 0.0), ("-", 0.0), ("-", 0.0)]
            else:
                top3 = [("Demo Mode (No Model)", 1.0), ("-", 0.0), ("-", 0.0)]
                self.detected_classes.append("Demo Traffic Sign")

            latency = (time.time() - t0) * 1000.0
            fps = 1000.0 / latency if latency > 0 else 0.0
            
            self.total_frames += 1
            self.total_latency += latency

            telemetry = {
                "latency": latency,
                "fps": fps,
                "npu_active": self.provider_name == "QNNExecutionProvider",
                "frames": self.total_frames
            }
            
            self.change_pixmap_signal.emit(frame, roi_32)
            self.update_pipeline_signal.emit(telemetry, top3)

    def trigger_session_end(self):
        elapsed = max(0.1, time.time() - self.start_timestamp)
        avg_latency = (self.total_latency / self.total_frames) if self.total_frames > 0 else 0.0
        avg_fps = (self.total_frames / elapsed) if elapsed > 0 else 0.0
        
        summary = {
            "total_frames": self.total_frames,
            "avg_latency": avg_latency,
            "avg_fps": avg_fps,
            "elapsed_time": elapsed,
            "class_counts": Counter(self.detected_classes),
            "hardware": "Snapdragon Hexagon NPU" if self.provider_name == "QNNExecutionProvider" else "CPU Fallback"
        }
        print(f"[DEBUG] Emitting Session Summary Report: {summary['total_frames']} frames evaluated.")
        self.session_ended_signal.emit(summary)

    def stop(self):
        self.running = False
        self.wait()


# ---------------------------------------------------------
# Professional Dashboard UI
# ---------------------------------------------------------
class SnapSignDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SnapSign ADAS: Qualcomm Edge NPU Visualizer")
        self.setGeometry(80, 80, 1150, 680)
        self.setStyleSheet("background-color: #0d0e12; color: #E0E6ED; font-family: 'Segoe UI', sans-serif;")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_dashboard_page()
        self.init_summary_page()

        self.thread = InferenceThread()
        self.thread.change_pixmap_signal.connect(self.update_video_and_roi)
        self.thread.update_pipeline_signal.connect(self.update_pipeline_view)
        self.thread.session_ended_signal.connect(self.show_summary_page)
        self.thread.start()

    def init_dashboard_page(self):
        dash_widget = QWidget()
        layout = QVBoxLayout(dash_widget)

        header = QHBoxLayout()
        title = QLabel("SnapSign ADAS Perception Dashboard")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #00E5FF;")
        
        self.hw_badge = QLabel("⚡ Acceleration: Hexagon NPU / QNN")
        self.hw_badge.setStyleSheet("background-color: #003B46; color: #00FFC8; padding: 6px 12px; border-radius: 12px; font-weight: bold;")
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.hw_badge)
        layout.addLayout(header)

        content = QHBoxLayout()

        # LEFT Panel
        left_box = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #1E232A; border-radius: 8px;")
        left_box.addWidget(self.video_label)

        controls = QHBoxLayout()
        self.btn_webcam = QPushButton("📷 Live Camera")
        self.btn_webcam.setStyleSheet(self.button_style())
        self.btn_webcam.clicked.connect(self.use_webcam)

        self.btn_video = QPushButton("🎞️ Load Video File")
        self.btn_video.setStyleSheet(self.button_style())
        self.btn_video.clicked.connect(self.load_video)

        self.btn_stop = QPushButton("⏹️ Stop & View Report")
        self.btn_stop.setStyleSheet(self.button_style("#D32F2F"))
        self.btn_stop.clicked.connect(self.stop_and_report)

        controls.addWidget(self.btn_webcam)
        controls.addWidget(self.btn_video)
        controls.addWidget(self.btn_stop)
        left_box.addLayout(controls)

        content.addLayout(left_box)

        # RIGHT Panel
        right_box = QVBoxLayout()

        roi_title = QLabel("Stage 1: Input Tensor Crop (32x32)")
        roi_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        right_box.addWidget(roi_title)

        roi_layout = QHBoxLayout()
        self.roi_label = QLabel()
        self.roi_label.setFixedSize(96, 96)
        self.roi_label.setStyleSheet("border: 2px solid #00E5FF; background-color: #000; border-radius: 4px;")
        
        roi_info = QLabel("Frame center cropped & normalized to 32x32 CHW tensor before sending to Snapdragon NPU.")
        roi_info.setWordWrap(True)
        roi_info.setStyleSheet("color: #8A99AD; font-size: 11px;")
        
        roi_layout.addWidget(self.roi_label)
        roi_layout.addWidget(roi_info)
        right_box.addLayout(roi_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #1E232A; margin: 10px 0;")
        right_box.addWidget(line)

        prob_title = QLabel("Stage 2: Model Inference Confidence")
        prob_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        right_box.addWidget(prob_title)

        self.top3_widgets = []
        for i in range(3):
            lbl = QLabel(f"Top {i+1}: --")
            lbl.setFont(QFont("Segoe UI", 10))
            bar = QProgressBar()
            bar.setFixedHeight(12)
            bar.setStyleSheet("""
                QProgressBar { border: none; background-color: #1E232A; border-radius: 6px; text-align: right; color: transparent;}
                QProgressBar::chunk { background-color: #00E5FF; border-radius: 6px; }
            """)
            right_box.addWidget(lbl)
            right_box.addWidget(bar)
            self.top3_widgets.append((lbl, bar))

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #1E232A; margin: 10px 0;")
        right_box.addWidget(line2)

        metrics_grid = QGridLayout()
        self.lbl_latency = self.create_metric_card(metrics_grid, "NPU Latency", "-- ms", 0, 0)
        self.lbl_fps = self.create_metric_card(metrics_grid, "Inference FPS", "--", 0, 1)
        self.lbl_frames = self.create_metric_card(metrics_grid, "Processed Frames", "0", 1, 0)
        self.lbl_status = self.create_metric_card(metrics_grid, "Status", "ACTIVE", 1, 1)

        right_box.addLayout(metrics_grid)
        right_box.addStretch()
        content.addLayout(right_box)

        layout.addLayout(content)
        self.stacked_widget.addWidget(dash_widget)

    def create_metric_card(self, grid, title_text, val_text, row, col):
        box = QWidget()
        box.setStyleSheet("background-color: #161A22; border-radius: 6px; padding: 8px;")
        v = QVBoxLayout(box)
        v.setContentsMargins(5, 5, 5, 5)
        
        t = QLabel(title_text)
        t.setStyleSheet("color: #8A99AD; font-size: 11px;")
        val = QLabel(val_text)
        val.setFont(QFont("Segoe UI", 13, QFont.Bold))
        val.setStyleSheet("color: #00FFC8;")
        
        v.addWidget(t)
        v.addWidget(val)
        grid.addWidget(box, row, col)
        return val

    def init_summary_page(self):
        summary_widget = QWidget()
        layout = QVBoxLayout(summary_widget)
        layout.setContentsMargins(40, 30, 40, 30)

        title = QLabel("📊 Snapdragon AI Edge - Session Diagnostics Report")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #00E5FF;")
        layout.addWidget(title)

        subtitle = QLabel("Inference run complete. Below is the summary of hardware performance and detected traffic sign classes.")
        subtitle.setStyleSheet("color: #8A99AD; margin-bottom: 15px;")
        layout.addWidget(subtitle)

        kpi_layout = QHBoxLayout()
        self.sum_kpi_frames = self.create_kpi_card(kpi_layout, "Total Frames", "0")
        self.sum_kpi_latency = self.create_kpi_card(kpi_layout, "Avg Latency", "0.0 ms")
        self.sum_kpi_fps = self.create_kpi_card(kpi_layout, "Avg FPS", "0.0")
        self.sum_kpi_hw = self.create_kpi_card(kpi_layout, "Target Hardware", "NPU")
        layout.addLayout(kpi_layout)

        table_title = QLabel("Traffic Sign Class Detections Breakdown")
        table_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        table_title.setStyleSheet("margin-top: 15px; color: #FFF;")
        layout.addWidget(table_title)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Class Name", "Occurrences", "% of Total Frames"])
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setStyleSheet("""
            QTableWidget { background-color: #161A22; gridline-color: #1E232A; border-radius: 8px; font-size: 13px; }
            QHeaderView::section { background-color: #1E232A; color: #00E5FF; padding: 8px; font-weight: bold; border: none; }
        """)
        layout.addWidget(self.summary_table)

        btn_restart = QPushButton("🔄 Start New Live Session")
        btn_restart.setStyleSheet(self.button_style("#00E5FF", color="#000"))
        btn_restart.setFixedHeight(45)
        btn_restart.setFont(QFont("Segoe UI", 11, QFont.Bold))
        btn_restart.clicked.connect(self.return_to_dashboard)
        layout.addWidget(btn_restart)

        self.stacked_widget.addWidget(summary_widget)

    def create_kpi_card(self, parent_layout, title, default_val):
        box = QWidget()
        box.setStyleSheet("background-color: #161A22; border: 1px solid #1E232A; border-radius: 8px; padding: 12px;")
        v = QVBoxLayout(box)
        t = QLabel(title)
        t.setStyleSheet("color: #8A99AD; font-size: 12px;")
        val = QLabel(default_val)
        val.setFont(QFont("Segoe UI", 16, QFont.Bold))
        val.setStyleSheet("color: #00FFC8;")
        v.addWidget(t)
        v.addWidget(val)
        parent_layout.addWidget(box)
        return val

    def update_video_and_roi(self, cv_frame, roi_crop):
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        q_frame = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_frame.scaled(640, 480, Qt.KeepAspectRatio)))

        rgb_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        rh, rw, rch = rgb_roi.shape
        q_roi = QImage(rgb_roi.data, rw, rh, rch * rw, QImage.Format_RGB888)
        self.roi_label.setPixmap(QPixmap.fromImage(q_roi.scaled(96, 96, Qt.KeepAspectRatio)))

    def update_pipeline_view(self, telemetry, top3):
        self.lbl_latency.setText(f"{telemetry['latency']:.1f} ms")
        self.lbl_fps.setText(f"{telemetry['fps']:.1f}")
        self.lbl_frames.setText(str(telemetry['frames']))

        for i, (label, prob) in enumerate(top3):
            lbl_widget, bar_widget = self.top3_widgets[i]
            lbl_widget.setText(f"{label} ({prob*100:.1f}%)")
            bar_widget.setValue(int(prob * 100))

    def show_summary_page(self, summary):
        # Update KPI Cards
        self.sum_kpi_frames.setText(str(summary["total_frames"]))
        self.sum_kpi_latency.setText(f"{summary['avg_latency']:.2f} ms")
        self.sum_kpi_fps.setText(f"{summary['avg_fps']:.1f}")
        self.sum_kpi_hw.setText(summary["hardware"])

        counts = summary["class_counts"]
        total = summary["total_frames"] or 1

        # Fallback if no classes detected
        if len(counts) == 0:
            self.summary_table.setRowCount(1)
            self.summary_table.setItem(0, 0, QTableWidgetItem("No Detections Registered"))
            self.summary_table.setItem(0, 1, QTableWidgetItem("0"))
            self.summary_table.setItem(0, 2, QTableWidgetItem("0%"))
        else:
            self.summary_table.setRowCount(len(counts))
            for row, (cls_name, count) in enumerate(counts.most_common()):
                pct = (count / total) * 100
                self.summary_table.setItem(row, 0, QTableWidgetItem(str(cls_name)))
                self.summary_table.setItem(row, 1, QTableWidgetItem(str(count)))
                self.summary_table.setItem(row, 2, QTableWidgetItem(f"{pct:.1f}%"))

        self.stacked_widget.setCurrentIndex(1)

    def return_to_dashboard(self):
        self.stacked_widget.setCurrentIndex(0)
        self.thread.set_source(0)

    def use_webcam(self):
        self.stacked_widget.setCurrentIndex(0)
        self.thread.set_source(0)

    def load_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Dashcam Video", "", "Video Files (*.mp4 *.avi *.mkv)", options=options)
        if file_name:
            self.stacked_widget.setCurrentIndex(0)
            self.thread.set_source(file_name)

    def stop_and_report(self):
        self.thread.trigger_session_end()

    def button_style(self, bg="#161A22", color="#FFF"):
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {color};
                padding: 8px 14px;
                border: 1px solid #1E232A;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #1E232A;
            }}
        """

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SnapSignDashboard()
    window.show()
    sys.exit(app.exec_())