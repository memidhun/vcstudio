import sys
import os
import subprocess
import zipfile
import shutil
import yaml
import cv2
import torch
import random
import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon

# Try to import ultralytics, prompt for installation if not found
try:
    from ultralytics import YOLO
    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False

# --- Utility Functions ---

def get_yolo_models():
    """Returns a list of common YOLOv8 models."""
    # You can expand this list based on ultralytics updates
    # or even try to fetch them dynamically if possible.
    return [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt',
        'yolov8l-seg.pt', 'yolov8x-seg.pt',
        # Add other versions or custom models if needed
    ]

def get_export_formats():
    """Returns a list of supported export formats."""
    return [
        'onnx', 'torchscript', 'coreml', 'saved_model', 'pb', 'tflite',
        'edgetpu', 'tfjs', 'paddle', 'ncnn', 'openvino'
    ]


def get_available_cameras():
    """Returns a list of available camera indices."""
    index = 0
    arr = []
    while index < 5:  # Check up to 5 cameras
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
    return arr

# --- Worker Threads ---

class InstallThread(QThread):
    """Thread to install ultralytics."""
    finished = pyqtSignal(bool, str)

    def run(self):
        try:
            self.log("Installing ultralytics...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'ultralytics'],
                check=True, capture_output=True, text=True
            )
            self.log("Ultralytics installed successfully.")
            global ULTRALYTICS_INSTALLED
            ULTRALYTICS_INSTALLED = True
            self.finished.emit(True, "Installation successful!")
        except subprocess.CalledProcessError as e:
            self.log(f"Installation failed: {e.stderr}")
            self.finished.emit(False, f"Installation failed: {e.stderr}")
        except Exception as e:
            self.log(f"An error occurred: {e}")
            self.finished.emit(False, f"An error occurred: {e}")

    def log(self, message):
        print(message) # Or emit a signal to log in GUI

class DataPrepThread(QThread):
    """Thread to prepare data: unzip, split, create yaml."""
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, zip_path, data_dir, train_pct=0.9):
        super().__init__()
        self.zip_path = zip_path
        self.data_dir = data_dir
        self.custom_data_dir = os.path.join(self.data_dir, "custom_data")
        self.final_data_dir = os.path.join(self.data_dir, "data")
        self.yaml_path = os.path.join(self.data_dir, "data.yaml")
        self.train_pct = train_pct

    def run(self):
        try:
            # 0. Cleanup previous attempts
            if os.path.exists(self.custom_data_dir): shutil.rmtree(self.custom_data_dir)
            if os.path.exists(self.final_data_dir): shutil.rmtree(self.final_data_dir)
            os.makedirs(self.custom_data_dir, exist_ok=True)
            os.makedirs(self.final_data_dir, exist_ok=True)
            self.progress.emit("Unzipping dataset...", 10)

            # 1. Unzip
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.custom_data_dir)
            self.progress.emit("Dataset unzipped.", 30)

            # --- Find images, labels, classes.txt ---
            # Search recursively for 'images', 'labels', 'classes.txt'
            images_src, labels_src, classes_src = None, None, None
            for root, dirs, files in os.walk(self.custom_data_dir):
                if 'images' in dirs and not images_src: images_src = os.path.join(root, 'images')
                if 'labels' in dirs and not labels_src: labels_src = os.path.join(root, 'labels')
                if 'classes.txt' in files and not classes_src: classes_src = os.path.join(root, 'classes.txt')
                if images_src and labels_src and classes_src: break # Found all

            if not images_src or not labels_src:
                self.finished.emit(False, "Could not find 'images' and/or 'labels' folders in the zip.")
                return
            if not classes_src:
                self.finished.emit(False, "Could not find 'classes.txt' in the zip.")
                return

            # --- Splitting Data ---
            self.progress.emit("Splitting data...", 50)
            all_images = [f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(all_images)
            split_idx = int(len(all_images) * self.train_pct)
            train_images = all_images[:split_idx]
            val_images = all_images[split_idx:]

            # --- Create final structure ---
            train_img_path = os.path.join(self.final_data_dir, 'train', 'images')
            train_lbl_path = os.path.join(self.final_data_dir, 'train', 'labels')
            val_img_path = os.path.join(self.final_data_dir, 'validation', 'images')
            val_lbl_path = os.path.join(self.final_data_dir, 'validation', 'labels')
            os.makedirs(train_img_path, exist_ok=True)
            os.makedirs(train_lbl_path, exist_ok=True)
            os.makedirs(val_img_path, exist_ok=True)
            os.makedirs(val_lbl_path, exist_ok=True)
            self.progress.emit("Copying files...", 70)

            # --- Copy files ---
            def copy_files(file_list, src_img, src_lbl, dst_img, dst_lbl):
                for img_file in file_list:
                    base_name = os.path.splitext(img_file)[0]
                    lbl_file = base_name + '.txt'
                    shutil.copy(os.path.join(src_img, img_file), dst_img)
                    if os.path.exists(os.path.join(src_lbl, lbl_file)):
                        shutil.copy(os.path.join(src_lbl, lbl_file), dst_lbl)

            copy_files(train_images, images_src, labels_src, train_img_path, train_lbl_path)
            copy_files(val_images, images_src, labels_src, val_img_path, val_lbl_path)
            self.progress.emit("Files copied.", 90)

            # --- Create data.yaml ---
            with open(classes_src, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]

            data_yaml_content = {
                'path': os.path.abspath(self.final_data_dir),
                'train': os.path.join('train', 'images'),
                'val': os.path.join('validation', 'images'),
                'nc': len(classes),
                'names': classes
            }
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data_yaml_content, f, sort_keys=False)
            self.progress.emit("data.yaml created.", 100)
            self.finished.emit(True, self.yaml_path)

        except Exception as e:
            self.finished.emit(False, f"Data preparation failed: {e}")

class TrainThread(QThread):
    """Thread to run YOLO training."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_name, data_yaml, epochs, imgsz, device):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.device = device

    def run(self):
        try:
            self.progress.emit(f"Starting training: {self.model_name}...")
            model = YOLO(self.model_name)
            
            # --- Training with callbacks for progress (basic) ---
            # Ultralytics callbacks might be complex to integrate directly for
            # live progress bars. We'll mainly report start/end/errors.
            # You could parse logs if running as subprocess, but API is cleaner.
            
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device=self.device,
                project="runs", # Saves to runs/detect/trainX
                name="gui_train"
            )
            
            # Find the best.pt path
            # The actual path might vary (train, train2, etc.)
            train_dirs = glob.glob("runs/detect/gui_train*")
            if not train_dirs:
                 train_dirs = glob.glob("runs/segment/gui_train*") # Check for segmentation

            if not train_dirs:
                self.finished.emit(False, "Could not find training results directory.")
                return

            latest_train_dir = max(train_dirs, key=os.path.getctime)
            best_pt_path = os.path.join(latest_train_dir, 'weights', 'best.pt')

            if os.path.exists(best_pt_path):
                 # Create my_model.pt
                 my_model_path = "my_model.pt"
                 shutil.copy(best_pt_path, my_model_path)
                 self.progress.emit(f"Training finished. Best model saved as {my_model_path}")
                 self.finished.emit(True, os.path.abspath(my_model_path))
            else:
                 self.finished.emit(False, f"Training completed, but 'best.pt' not found in {latest_train_dir}.")

        except Exception as e:
            self.progress.emit(f"Training failed: {e}")
            self.finished.emit(False, f"Training failed: {e}")


class InferenceThread(QThread):
    """Thread for live inference."""
    frame_ready = pyqtSignal(QImage)
    finished = pyqtSignal(str)

    def __init__(self, model_path, source_type, source_path, device):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type
        self.source_path = source_path
        self.device = device
        self._is_running = True

    def run(self):
        try:
            model = YOLO(self.model_path)
            
            if self.source_type == 'webcam':
                cap = cv2.VideoCapture(int(self.source_path))
                if not cap.isOpened():
                    self.finished.emit("Error: Could not open webcam.")
                    return
            elif self.source_type == 'video':
                cap = cv2.VideoCapture(self.source_path)
                if not cap.isOpened():
                    self.finished.emit("Error: Could not open video file.")
                    return
            elif self.source_type == 'image':
                 img = cv2.imread(self.source_path)
                 if img is None:
                     self.finished.emit("Error: Could not open image file.")
                     return
                 results = model.predict(img, device=self.device)
                 annotated_frame = results[0].plot()
                 self.emit_frame(annotated_frame)
                 self.finished.emit("Image inference complete.")
                 return # Done for images
            else:
                self.finished.emit("Error: Invalid source type.")
                return

            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame, device=self.device, verbose=False)
                annotated_frame = results[0].plot()
                self.emit_frame(annotated_frame)

            cap.release()
            self.finished.emit("Inference stopped.")

        except Exception as e:
            self.finished.emit(f"Inference error: {e}")

    def emit_frame(self, frame):
        """Converts CV2 frame to QImage and emits signal."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.frame_ready.emit(qt_image.scaled(640, 480, Qt.KeepAspectRatio))

    def stop(self):
        self._is_running = False

class ExportThread(QThread):
    """Thread to export model."""
    finished = pyqtSignal(bool, str)

    def __init__(self, model_path, export_format):
        super().__init__()
        self.model_path = model_path
        self.export_format = export_format

    def run(self):
        try:
            model = YOLO(self.model_path)
            exported_path = model.export(format=self.export_format)
            self.finished.emit(True, f"Model exported to {exported_path}")
        except Exception as e:
            self.finished.emit(False, f"Export failed: {e}")

# --- Main Application Window ---

class YoloTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Training & Deployment GUI")
        self.setGeometry(100, 100, 800, 700)
        self.setWindowIcon(QIcon()) # Add an icon if you have one

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Variables
        self.dataset_zip_path = None
        self.data_yaml_path = None
        self.trained_model_path = None
        self.inference_thread = None

        # Scroll Area for long content
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.main_layout.addWidget(self.scroll)

        # --- Sections ---
        self.create_setup_section()
        self.create_data_section()
        self.create_model_section()
        self.create_train_section()
        self.create_deploy_section()
        self.create_log_section()

        self.scroll_layout.addStretch() # Pushes content up
        self.scroll.setWidget(self.scroll_content)

        # Status Bar
        self.statusBar().showMessage("Ready. Please follow the steps.")
        self.check_ultralytics()


    def create_setup_section(self):
        group = QGroupBox("1. Setup")
        layout = QVBoxLayout()

        # Install Ultralytics
        self.install_label = QLabel("Ultralytics Status: Unknown")
        self.install_button = QPushButton("Install/Check Ultralytics")
        self.install_button.clicked.connect(self.install_ultralytics)
        layout.addWidget(self.install_label)
        layout.addWidget(self.install_button)

        # Device Selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Select Compute Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("cpu")
        if torch.cuda.is_available():
            self.device_combo.addItem(f"cuda:0 ({torch.cuda.get_device_name(0)})")
            self.device_combo.setCurrentIndex(1)
        else:
             self.device_combo.setCurrentIndex(0)
             self.device_combo.setEnabled(False) # No GPU, only CPU

        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_data_section(self):
        group = QGroupBox("2. Dataset Preparation")
        layout = QVBoxLayout()

        self.data_zip_label = QLabel("No dataset selected.")
        load_button = QPushButton("Load Custom Dataset (.zip)")
        load_button.clicked.connect(self.load_dataset)
        layout.addWidget(self.data_zip_label)
        layout.addWidget(load_button)

        self.data_prep_progress = QProgressBar()
        self.data_prep_status = QLabel("Status: Waiting for dataset...")
        layout.addWidget(self.data_prep_progress)
        layout.addWidget(self.data_prep_status)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_model_section(self):
        group = QGroupBox("3. Model Selection")
        layout = QVBoxLayout()

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Choose YOLO Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_yolo_models())
        self.model_combo.setCurrentText("yolov8s.pt")
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_train_section(self):
        group = QGroupBox("4. Training")
        layout = QVBoxLayout()

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(50) # Default
        params_layout.addWidget(self.epochs_spinbox)

        params_layout.addWidget(QLabel("Image Size (px):"))
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 2048)
        self.imgsz_spinbox.setValue(640) # Default
        self.imgsz_spinbox.setSingleStep(32)
        params_layout.addWidget(self.imgsz_spinbox)
        layout.addLayout(params_layout)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False) # Enable after data prep
        layout.addWidget(self.train_button)

        self.train_status_label = QLabel("Status: Waiting for configuration...")
        layout.addWidget(self.train_status_label)

        group.setLayout(layout)
        self.scroll_layout.addWidget(group)

    def create_deploy_section(self):
        group = QGroupBox("5. Deployment & Inference")
        layout = QVBoxLayout()
        
        # --- Inference Part ---
        inference_group = QGroupBox("Live Inference / Test")
        inf_layout = QVBoxLayout()

        self.inference_status = QLabel("Load a trained model (my_model.pt) first.")
        self.load_trained_model_button = QPushButton("Load Trained Model (.pt)")
        self.load_trained_model_button.clicked.connect(self.load_trained_model)
        inf_layout.addWidget(self.inference_status)
        inf_layout.addWidget(self.load_trained_model_button)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setText("Webcam/Video Feed Will Appear Here")
        inf_layout.addWidget(self.video_label)

        inf_controls = QHBoxLayout()
        self.webcam_combo = QComboBox()
        self.webcam_combo.addItems([f"Cam {i}" for i in get_available_cameras()])
        self.start_webcam_button = QPushButton("Start Webcam")
        self.start_webcam_button.clicked.connect(self.start_webcam_inference)
        self.load_video_button = QPushButton("Load Video/Image")
        self.load_video_button.clicked.connect(self.start_file_inference)
        self.stop_inference_button = QPushButton("Stop")
        self.stop_inference_button.clicked.connect(self.stop_inference)
        self.start_webcam_button.setEnabled(False)
        self.load_video_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        inf_controls.addWidget(self.webcam_combo)
        inf_controls.addWidget(self.start_webcam_button)
        inf_controls.addWidget(self.load_video_button)
        inf_controls.addWidget(self.stop_inference_button)
        inf_layout.addLayout(inf_controls)
        inference_group.setLayout(inf_layout)
        layout.addWidget(inference_group)

        # --- Export Part ---
        export_group = QGroupBox("Model Conversion")
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Export to:"))
        self.export_combo = QComboBox()
        self.export_combo.addItems(get_export_formats())
        self.export_button = QPushButton("Convert Model")
        self.export_button.clicked.connect(self.export_model)
        self.export_button.setEnabled(False) # Enable when model loaded

        exp_layout.addWidget(self.export_combo)
        exp_layout.addWidget(self.export_button)
        export_group.setLayout(exp_layout)
        layout.addWidget(export_group)


        group.setLayout(layout)
        self.scroll_layout.addWidget(group)


    def create_log_section(self):
        group = QGroupBox("Logs & Status")
        layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(150)
        layout.addWidget(self.log_text)
        group.setLayout(layout)
        self.main_layout.addWidget(group) # Put logs at bottom, outside scroll


    def log(self, message):
        """Appends a message to the log area and status bar."""
        self.log_text.append(message)
        self.statusBar().showMessage(message, 5000) # Show for 5 seconds
        print(message) # Also print to console for debugging

    # --- Actions & Slots ---

    def check_ultralytics(self):
        if ULTRALYTICS_INSTALLED:
            self.install_label.setText("Ultralytics Status: Installed ✔️")
            self.install_button.setEnabled(False)
            self.log("Ultralytics is ready.")
        else:
            self.install_label.setText("Ultralytics Status: Not Installed ❌")
            self.log("Ultralytics is not installed. Please install it.")
            # Disable other buttons until installed
            self.train_button.setEnabled(False)
            self.start_webcam_button.setEnabled(False)
            self.load_video_button.setEnabled(False)
            self.export_button.setEnabled(False)


    def install_ultralytics(self):
        self.install_button.setEnabled(False)
        self.log("Starting Ultralytics installation...")
        self.install_thread = InstallThread()
        self.install_thread.finished.connect(self.on_install_finished)
        self.install_thread.start()

    def on_install_finished(self, success, message):
        self.log(message)
        self.check_ultralytics()
        if not success:
            self.install_button.setEnabled(True) # Allow retry

    def load_dataset(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Dataset Zip File", "", "Zip Files (*.zip)", options=options)
        if fileName:
            self.dataset_zip_path = fileName
            self.data_zip_label.setText(f"Selected: {os.path.basename(fileName)}")
            self.log(f"Dataset selected: {fileName}")
            self.start_data_prep()

    def start_data_prep(self):
        if not self.dataset_zip_path:
            self.log("Error: No dataset selected.")
            return

        self.log("Starting data preparation...")
        self.data_prep_status.setText("Status: Preparing...")
        self.data_prep_progress.setValue(0)
        
        # Create a temp dir for data processing
        self.work_dir = "yolo_gui_workdir"
        os.makedirs(self.work_dir, exist_ok=True)

        self.data_prep_thread = DataPrepThread(self.dataset_zip_path, self.work_dir)
        self.data_prep_thread.progress.connect(self.update_data_prep_progress)
        self.data_prep_thread.finished.connect(self.on_data_prep_finished)
        self.data_prep_thread.start()


    def update_data_prep_progress(self, message, value):
        self.data_prep_status.setText(f"Status: {message}")
        self.data_prep_progress.setValue(value)

    def on_data_prep_finished(self, success, message):
        if success:
            self.data_yaml_path = message
            self.log(f"Data ready. Config file at: {self.data_yaml_path}")
            self.data_prep_status.setText("Status: Data Ready ✔️")
            self.train_button.setEnabled(ULTRALYTICS_INSTALLED) # Enable training
        else:
            self.log(f"Data preparation failed: {message}")
            self.data_prep_status.setText(f"Status: Failed ❌ ({message})")
            self.train_button.setEnabled(False)

    def start_training(self):
        if not self.data_yaml_path:
            self.log("Error: Data is not prepared.")
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics is not installed.")
            return

        self.train_button.setEnabled(False)
        self.train_status_label.setText("Status: Training in progress...")
        self.log("Starting training...")

        model = self.model_combo.currentText()
        epochs = self.epochs_spinbox.value()
        imgsz = self.imgsz_spinbox.value()
        device_text = self.device_combo.currentText()
        device = 'cpu' if device_text == 'cpu' else 0 # Ultralytics takes 0 for first GPU

        self.train_thread = TrainThread(model, self.data_yaml_path, epochs, imgsz, device)
        self.train_thread.progress.connect(self.log) # Log progress messages
        self.train_thread.finished.connect(self.on_training_finished)
        self.train_thread.start()

    def on_training_finished(self, success, message):
        self.train_button.setEnabled(True)
        if success:
            self.trained_model_path = message
            self.train_status_label.setText(f"Status: Training Complete ✔️ Model: {os.path.basename(message)}")
            self.log(f"Training finished. Model saved at {message}")
            self.load_trained_model(self.trained_model_path) # Auto-load it
        else:
            self.train_status_label.setText(f"Status: Training Failed ❌ ({message})")
            self.log(f"Training failed: {message}")

    def load_trained_model(self, model_path=None):
        if not model_path:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getOpenFileName(self, "Select Trained Model File", "", "PyTorch Models (*.pt)", options=options)
            if not fileName:
                return
            model_path = fileName
            
        self.trained_model_path = model_path
        self.log(f"Loaded trained model: {self.trained_model_path}")
        self.inference_status.setText(f"Model Loaded: {os.path.basename(self.trained_model_path)}")
        # Enable inference and export buttons
        self.start_webcam_button.setEnabled(ULTRALYTICS_INSTALLED)
        self.load_video_button.setEnabled(ULTRALYTICS_INSTALLED)
        self.export_button.setEnabled(ULTRALYTICS_INSTALLED)


    def start_inference(self, source_type, source_path):
        if not self.trained_model_path:
            self.log("Error: No trained model loaded for inference.")
            return
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Inference already running. Stop it first.")
            return

        self.log(f"Starting inference on {source_type}: {source_path}")
        device_text = self.device_combo.currentText()
        device = 'cpu' if device_text == 'cpu' else 0

        self.inference_thread = InferenceThread(self.trained_model_path, source_type, source_path, device)
        self.inference_thread.frame_ready.connect(self.update_video_frame)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()
        self.stop_inference_button.setEnabled(True)
        self.start_webcam_button.setEnabled(False)
        self.load_video_button.setEnabled(False)


    def start_webcam_inference(self):
        cam_text = self.webcam_combo.currentText() # "Cam 0"
        cam_index = cam_text.split()[-1]
        self.start_inference('webcam', cam_index)

    def start_file_inference(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video or Image File", "",
                                                  "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png)", options=options)
        if fileName:
            if fileName.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.start_inference('image', fileName)
            else:
                 self.start_inference('video', fileName)

    def stop_inference(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Stopping inference...")
            self.inference_thread.stop()
            self.stop_inference_button.setEnabled(False)

    def update_video_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def on_inference_finished(self, message):
        self.log(f"Inference finished: {message}")
        self.video_label.setText("Inference Stopped / Finished")
        self.stop_inference_button.setEnabled(False)
        self.start_webcam_button.setEnabled(bool(self.trained_model_path))
        self.load_video_button.setEnabled(bool(self.trained_model_path))
        self.inference_thread = None

    def export_model(self):
        if not self.trained_model_path:
            self.log("Error: No trained model loaded to export.")
            return

        export_format = self.export_combo.currentText()
        self.log(f"Starting export to {export_format}...")
        self.export_button.setEnabled(False)

        self.export_thread = ExportThread(self.trained_model_path, export_format)
        self.export_thread.finished.connect(self.on_export_finished)
        self.export_thread.start()

    def on_export_finished(self, success, message):
        self.log(message)
        self.export_button.setEnabled(True)


    def closeEvent(self, event):
        """Ensure threads are stopped when closing."""
        self.stop_inference()
        # You might need to add similar stop mechanisms for other threads
        # if they can run for very long or need graceful shutdown.
        event.accept()

# --- Run the App ---

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YoloTrainerGUI()
    window.show()
    sys.exit(app.exec_())