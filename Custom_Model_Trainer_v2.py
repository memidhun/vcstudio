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
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QScrollArea, QSizePolicy,
    QStackedWidget, QFrame,QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont

# Try to import ultralytics, prompt for installation if not found
try:
    from ultralytics import YOLO
    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False

# --- Utility Functions ---

def get_yolo_models():
    return [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt',
        'yolov8l-seg.pt', 'yolov8x-seg.pt',
    ]

def get_export_formats():
    return [
        'onnx', 'torchscript', 'coreml', 'saved_model', 'pb', 'tflite',
        'edgetpu', 'tfjs', 'paddle', 'ncnn', 'openvino'
    ]

def get_available_cameras():
    index = 0
    arr = []
    while index < 5:  # Check up to 5 cameras
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # CAP_DSHOW can be faster on Windows
        if cap.isOpened():
            if cap.read()[0]:
                arr.append(index)
            cap.release()
        index += 1
    return arr

# --- Worker Threads (Largely unchanged, minor adjustments for new signals if any) ---

class InstallThread(QThread):
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def run(self):
        try:
            self.log_message.emit("Installing ultralytics...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'ultralytics'],
                check=True, capture_output=True, text=True
            )
            self.log_message.emit("Ultralytics installed successfully.")
            global ULTRALYTICS_INSTALLED
            ULTRALYTICS_INSTALLED = True
            self.finished.emit(True, "Installation successful!")
        except subprocess.CalledProcessError as e:
            error_msg = f"Installation failed: {e.stderr or e.stdout or str(e)}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, error_msg)
        except Exception as e:
            error_msg = f"An error occurred during installation: {e}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, error_msg)

class DataPrepThread(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, zip_path, data_dir, train_pct=0.9):
        super().__init__()
        self.zip_path = zip_path
        self.data_dir = data_dir
        self.custom_data_dir = os.path.join(self.data_dir, "custom_data_extracted")
        self.final_data_dir = os.path.join(self.data_dir, "dataset_formatted")
        self.yaml_path = os.path.join(self.final_data_dir, "data.yaml") # Place yaml inside formatted dataset
        self.train_pct = train_pct

    def run(self):
        try:
            self.log_message.emit(f"Starting data preparation for {os.path.basename(self.zip_path)}...")
            if os.path.exists(self.custom_data_dir): shutil.rmtree(self.custom_data_dir)
            if os.path.exists(self.final_data_dir): shutil.rmtree(self.final_data_dir)
            os.makedirs(self.custom_data_dir, exist_ok=True)
            os.makedirs(self.final_data_dir, exist_ok=True)
            self.progress.emit("Unzipping dataset...", 10)

            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.custom_data_dir)
            self.log_message.emit("Dataset unzipped.")
            self.progress.emit("Dataset unzipped.", 30)

            images_src, labels_src, classes_file_path = None, None, None
            
            # More robust search for dataset structure
            # Common structures:
            # 1. zip -> images/, labels/, classes.txt
            # 2. zip -> dataset_name/ -> images/, labels/, classes.txt
            # 3. zip -> train/images, train/labels, val/images, val/labels, data.yaml (Roboflow like)

            possible_classes_files = glob.glob(os.path.join(self.custom_data_dir, '**', 'classes.txt'), recursive=True)
            if possible_classes_files:
                classes_file_path = possible_classes_files[0]
                self.log_message.emit(f"Found classes.txt at {classes_file_path}")
            
            # Try to find 'images' and 'labels' at various depths
            for i in range(3): # Check root, one level deep, two levels deep
                search_path = self.custom_data_dir
                for _ in range(i):
                    dirs_in_search_path = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))]
                    if len(dirs_in_search_path) == 1: # If only one subdir, assume it's a container
                        search_path = os.path.join(search_path, dirs_in_search_path[0])
                    else: # If multiple or no subdirs, search in current path
                        break
                
                possible_images_dirs = glob.glob(os.path.join(search_path, '**', 'images'), recursive=True)
                possible_labels_dirs = glob.glob(os.path.join(search_path, '**', 'labels'), recursive=True)

                if possible_images_dirs: images_src = possible_images_dirs[0]
                if possible_labels_dirs: labels_src = possible_labels_dirs[0]

                if images_src and labels_src:
                    self.log_message.emit(f"Found images in: {images_src}")
                    self.log_message.emit(f"Found labels in: {labels_src}")
                    break
            
            if not images_src or not os.path.isdir(images_src):
                self.finished.emit(False, "Could not find 'images' folder. Please ensure it's named 'images'.")
                return
            if not labels_src or not os.path.isdir(labels_src):
                self.finished.emit(False, "Could not find 'labels' folder. Please ensure it's named 'labels'.")
                return
            if not classes_file_path or not os.path.isfile(classes_file_path):
                 # Try to infer classes from label files if classes.txt is missing
                self.log_message.emit("classes.txt not found. Will attempt to infer classes from label files if possible or require manual creation.")
                # For simplicity, we'll still require classes.txt for now. This part can be expanded.
                self.finished.emit(False, "Could not find 'classes.txt'. It must be present in the zip.")
                return


            self.progress.emit("Splitting data...", 50)
            all_images = sorted([f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Ensure corresponding label files exist for images before adding to list
            valid_images = []
            for img_file in all_images:
                base_name = os.path.splitext(img_file)[0]
                lbl_file = base_name + '.txt'
                if os.path.exists(os.path.join(labels_src, lbl_file)):
                    valid_images.append(img_file)
                else:
                    self.log_message.emit(f"Warning: Label file for {img_file} not found. Skipping this image.")
            
            if not valid_images:
                self.finished.emit(False, "No valid image-label pairs found.")
                return

            random.shuffle(valid_images)
            split_idx = int(len(valid_images) * self.train_pct)
            train_images = valid_images[:split_idx]
            val_images = valid_images[split_idx:]

            train_img_path = os.path.join(self.final_data_dir, 'train', 'images')
            train_lbl_path = os.path.join(self.final_data_dir, 'train', 'labels')
            val_img_path = os.path.join(self.final_data_dir, 'val', 'images') # YOLO expects 'val' or 'valid'
            val_lbl_path = os.path.join(self.final_data_dir, 'val', 'labels')
            
            os.makedirs(train_img_path, exist_ok=True)
            os.makedirs(train_lbl_path, exist_ok=True)
            os.makedirs(val_img_path, exist_ok=True)
            os.makedirs(val_lbl_path, exist_ok=True)
            self.progress.emit("Copying files...", 70)

            def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
                for img_file in file_list:
                    base_name = os.path.splitext(img_file)[0]
                    lbl_file = base_name + '.txt'
                    shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
                    shutil.copy(os.path.join(src_lbl_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))

            copy_files(train_images, images_src, labels_src, train_img_path, train_lbl_path)
            copy_files(val_images, images_src, labels_src, val_img_path, val_lbl_path)
            self.log_message.emit("Train/validation files copied.")
            self.progress.emit("Files copied.", 90)

            with open(classes_file_path, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]

            data_yaml_content = {
                'path': os.path.abspath(self.final_data_dir), # Absolute path to dataset root
                'train': os.path.join('train', 'images'),      # Relative path from 'path'
                'val': os.path.join('val', 'images'),        # Relative path from 'path'
                'nc': len(classes),
                'names': classes
            }
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)
            self.log_message.emit(f"data.yaml created at {self.yaml_path}")
            self.progress.emit("data.yaml created.", 100)
            self.finished.emit(True, self.yaml_path)

        except Exception as e:
            self.log_message.emit(f"Data preparation error: {e}")
            self.finished.emit(False, f"Data preparation failed: {e}")


class TrainThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # True/False for success, str for message or model path

    def __init__(self, model_name, data_yaml, epochs, imgsz, device, project_name="yolo_gui_runs"):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.device = device
        self.project_name = project_name
        self.run_name = "train_custom"


    def run(self):
        try:
            self.progress_update.emit(f"Initializing YOLO model: {self.model_name} for training.")
            model = YOLO(self.model_name)
            self.progress_update.emit(f"Starting training on device: {self.device if self.device is not None else 'cpu'}")
            self.progress_update.emit(f"Dataset: {self.data_yaml}, Epochs: {self.epochs}, Image Size: {self.imgsz}")

            # Deleting previous run if it exists to get a clean 'train_custom' dir
            train_run_dir_pattern = os.path.join(self.project_name, "detect", self.run_name + "*")
            segment_run_dir_pattern = os.path.join(self.project_name, "segment", self.run_name + "*")
            
            for pattern in [train_run_dir_pattern, segment_run_dir_pattern]:
                existing_runs = glob.glob(pattern)
                for run_dir in existing_runs:
                    if os.path.basename(run_dir) == self.run_name: # exact match
                         self.progress_update.emit(f"Deleting existing run directory: {run_dir}")
                         shutil.rmtree(run_dir)


            # For detailed progress, Ultralytics has callbacks, but they are complex to directly
            # integrate into Qt signals per epoch smoothly without modifying Ultralytics source.
            # We'll rely on console output parsing or simpler signals.
            # For now, we signal start and end.
            
            # Simplified device handling for ultralytics
            device_arg = self.device if self.device not in ['cpu', None] else 'cpu'


            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device=device_arg,
                project=self.project_name, # Saves to project_name/detect/train_custom
                name=self.run_name,
                exist_ok=False # Ensures it creates a new directory or errors if 'train_custom' (without number) exists
            )
            
            # Find the latest training directory path (e.g., runs/detect/train, runs/detect/train2, etc.)
            # The actual path will be project_name/detect/train_custom or project_name/segment/train_custom
            
            task_type = "detect" if "seg" not in self.model_name else "segment"
            latest_train_dir = os.path.join(self.project_name, task_type, self.run_name) # This should be the one just created

            if not os.path.exists(latest_train_dir):
                 # Fallback if naming convention changed or failed
                train_dirs = glob.glob(os.path.join(self.project_name, task_type, self.run_name + "*"))
                if not train_dirs:
                    self.finished.emit(False, "Could not find training results directory.")
                    return
                latest_train_dir = max(train_dirs, key=os.path.getctime)


            best_pt_path = os.path.join(latest_train_dir, 'weights', 'best.pt')

            if os.path.exists(best_pt_path):
                # Copy best.pt to a known location, e.g., application's root or a 'trained_models' folder
                output_model_dir = "trained_models_gui"
                os.makedirs(output_model_dir, exist_ok=True)
                
                # Create a unique name for the copied model
                base_model_name = os.path.splitext(os.path.basename(self.model_name))[0]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_model_name = f"{base_model_name}_custom_{timestamp}.pt"
                final_model_path = os.path.join(output_model_dir, final_model_name)
                
                shutil.copy(best_pt_path, final_model_path)
                self.progress_update.emit(f"Training finished. Best model saved as {final_model_path}")
                self.finished.emit(True, os.path.abspath(final_model_path))
            else:
                self.progress_update.emit(f"Training completed, but 'best.pt' not found in {latest_train_dir}/weights.")
                self.finished.emit(False, f"Training completed, but 'best.pt' not found in {latest_train_dir}/weights.")

        except Exception as e:
            self.progress_update.emit(f"Training error: {e}")
            self.finished.emit(False, f"Training failed: {e}")


class InferenceThread(QThread):
    frame_ready = pyqtSignal(QImage, float, int) # Image, FPS, Object Count
    finished = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, model_path, source_type, source_path, device, confidence_threshold=0.25):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type
        self.source_path = source_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._is_running = True
        self.model = None
        self.cap = None
        self.prev_time = 0
        self.fps = 0

    def run(self):
        try:
            self.log_message.emit(f"Loading model for inference: {self.model_path} on device: {self.device}")
            self.model = YOLO(self.model_path)
            self.log_message.emit("Model loaded successfully.")

            if self.source_type == 'webcam':
                try:
                    cam_idx = int(self.source_path)
                    self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.finished.emit(f"Error: Could not open webcam {cam_idx}.")
                        return
                    self.log_message.emit(f"Webcam {cam_idx} opened.")
                except ValueError:
                    self.finished.emit(f"Error: Invalid webcam index '{self.source_path}'.")
                    return
            elif self.source_type == 'video':
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.finished.emit(f"Error: Could not open video file: {self.source_path}")
                    return
                self.log_message.emit(f"Video file opened: {self.source_path}")
            elif self.source_type == 'image':
                img = cv2.imread(self.source_path)
                if img is None:
                    self.finished.emit(f"Error: Could not open image file: {self.source_path}")
                    return
                self.log_message.emit(f"Image file opened: {self.source_path}")
                
                results = self.model.predict(img, device=self.device, conf=self.confidence_threshold, verbose=False)
                annotated_frame = results[0].plot()
                object_count = len(results[0].boxes)
                self.emit_frame(annotated_frame, 0, object_count) # FPS is 0 for single image
                self.finished.emit("Image inference complete.")
                return
            else:
                self.finished.emit("Error: Invalid source type.")
                return

            self.prev_time = time.time()

            while self._is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message.emit("End of video stream or cannot read frame.")
                    break
                
                current_time = time.time()
                elapsed_time = current_time - self.prev_time
                self.fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                self.prev_time = current_time

                results = self.model.predict(frame, device=self.device, conf=self.confidence_threshold, verbose=False)
                annotated_frame = results[0].plot() # plot() returns a BGR numpy array
                object_count = len(results[0].boxes)
                
                self.emit_frame(annotated_frame, self.fps, object_count)
                QThread.msleep(10) # Small delay to prevent GUI freeze, adjust as needed

            if self.cap:
                self.cap.release()
            self.finished.emit("Inference stopped.")

        except Exception as e:
            self.log_message.emit(f"Inference error: {e}")
            self.finished.emit(f"Inference error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.log_message.emit("Inference thread resources released.")


    def emit_frame(self, frame_cv, fps, obj_count):
        rgb_image = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.frame_ready.emit(qt_image.scaled(640, 480, Qt.KeepAspectRatio), fps, obj_count)

    def stop(self):
        self.log_message.emit("Stopping inference thread...")
        self._is_running = False


class ExportThread(QThread):
    finished = pyqtSignal(bool, str) # Success, Message/Path
    log_message = pyqtSignal(str)

    def __init__(self, model_path, export_format):
        super().__init__()
        self.model_path = model_path
        self.export_format = export_format

    def run(self):
        try:
            self.log_message.emit(f"Loading model for export: {self.model_path}")
            model = YOLO(self.model_path)
            self.log_message.emit(f"Exporting model to {self.export_format} format...")
            # Ultralytics export might save next to the model or in 'runs/export'
            # Let's try to make the output path more predictable if possible
            output_dir = "exported_models"
            os.makedirs(output_dir, exist_ok=True)
            model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
            
            # Note: Ultralytics' export() often determines filename itself.
            # We might need to find the exported file.
            # For now, we rely on its default behavior and report the path it returns.
            exported_path = model.export(format=self.export_format,
                                         # imgsz=640 # Optional: specify image size for export
                                         ) 
            
            # Try to move the exported model to our designated directory if it's not already there
            # And if exported_path is a file (not a directory like for saved_model)
            if os.path.isfile(exported_path) and output_dir not in exported_path:
                final_exported_filename = f"{model_basename}.{self.export_format}"
                if self.export_format == 'saved_model': # TF saved_model is a directory
                    final_exported_path = os.path.join(output_dir, model_basename + "_saved_model")
                    if os.path.exists(final_exported_path): shutil.rmtree(final_exported_path)
                    shutil.move(exported_path, final_exported_path) # move the whole dir
                elif self.export_format == 'pb': # TF GraphDef
                     final_exported_filename = f"{model_basename}_frozen.pb"
                     final_exported_path = os.path.join(output_dir, final_exported_filename)
                     shutil.move(exported_path, final_exported_path)
                else:
                    # For other formats like onnx, tflite etc.
                    final_exported_path = os.path.join(output_dir, f"{model_basename}.{self.export_format if self.export_format != 'torchscript' else 'torchscript.pt'}")
                    shutil.move(exported_path, final_exported_path)
                exported_path = os.path.abspath(final_exported_path)

            self.log_message.emit(f"Model successfully exported to: {exported_path}")
            self.finished.emit(True, f"Model exported to {exported_path}")
        except Exception as e:
            self.log_message.emit(f"Export failed: {e}")
            self.finished.emit(False, f"Export failed: {e}")


# --- Main Application Window ---

class ModernYoloGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionCraft Studio")
        self.setGeometry(50, 50, 1000, 750) # Increased default size
        self.setWindowIcon(QIcon("path/to/your/app_icon.png")) # Add your app icon

        # Variables
        self.dataset_zip_path = None
        self.data_yaml_path = None
        self.trained_model_path_for_inference = None # Model loaded specifically for inference tab
        self.inference_thread = None
        self.current_theme = "light" # or "dark"
        self.work_dir = "yolo_gui_workspace" # Centralized working directory
        os.makedirs(self.work_dir, exist_ok=True)


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.setSpacing(0)

        # Page Container
        self.stacked_widget = QStackedWidget()

        # Create Pages
        self.home_page = self.create_home_page()
        self.data_page = self.create_data_page()
        self.train_page = self.create_train_page()
        self.deploy_page = self.create_deploy_page()
        self.settings_page = self.create_settings_page()

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.data_page)
        self.stacked_widget.addWidget(self.train_page)
        self.stacked_widget.addWidget(self.deploy_page)
        self.stacked_widget.addWidget(self.settings_page)

        # Navigation Bar
        self.nav_bar = self.create_navigation_bar()

        # Log Area (shared at the bottom)
        self.log_group = QGroupBox("Console Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(100) # Adjust height as needed
        log_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_layout)

        self.main_layout.addWidget(self.stacked_widget)
        self.main_layout.addWidget(self.nav_bar)
        self.main_layout.addWidget(self.log_group) # Log group below nav bar

        # Apply Stylesheet
        self.apply_stylesheet(self.current_theme)

        # Initial setup
        self.log("Application started. Welcome to VisionCraft Studio!")
        self.update_ultralytics_status_display()
        if not ULTRALYTICS_INSTALLED:
            self.log("Ultralytics library is not detected. Please install it via the Settings page.")
        self.update_inference_controls_state() # Disable inference if no model loaded


    def create_navigation_bar(self):
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0,0,0,0)
        nav_layout.setSpacing(0) # No space between nav buttons

        # Icons are placeholders, replace with actual paths to good quality icons
        # Recommended size for icons: 24x24 or 32x32
        # Example using QIcon.fromTheme for standard icons if available, else use file paths
        nav_buttons_data = [
            {"text": "Home", "icon": "path/to/home_icon.png", "page_index": 0},
            {"text": "Data", "icon": "path/to/data_icon.png", "page_index": 1},
            {"text": "Train", "icon": "path/to/train_icon.png", "page_index": 2},
            {"text": "Deploy", "icon": "path/to/deploy_icon.png", "page_index": 3},
            {"text": "Settings", "icon": "path/to/settings_icon.png", "page_index": 4},
        ]

        self.nav_buttons = []
        for item in nav_buttons_data:
            button = QPushButton(QIcon(item["icon"]), f" {item['text']}") # Space for text padding
            button.setIconSize(QSize(24,24))
            button.setCheckable(True)
            button.setProperty("class", "navButton")
            button.clicked.connect(lambda checked, index=item["page_index"]: self.switch_page(index))
            nav_layout.addWidget(button)
            self.nav_buttons.append(button)
        
        self.nav_buttons[0].setChecked(True) # Set Home as default active
        nav_widget.setFixedHeight(50) # Adjust height of nav bar
        return nav_widget

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.log(f"Switched to {self.nav_buttons[index].text().strip()} page.")
    ######################################################################################################################
    def select_dataset_zip(self): # 
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly # Suggest read-only access
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset Zip File",
            "",  # Start directory (empty for last used or default)
            "Zip Files (*.zip)",
            options=options
        )
        if fileName:
            self.dataset_zip_path = fileName
            # Assuming self.data_zip_label is defined in create_data_page
            if hasattr(self, 'data_zip_label') and self.data_zip_label:
                self.data_zip_label.setText(f"Selected: {os.path.basename(fileName)}")
            self.log(f"Dataset ZIP selected: {fileName}")
            self.data_yaml_path = None  # Reset previous YAML path
            if hasattr(self, 'data_yaml_path_label') and self.data_yaml_path_label:
                 self.data_yaml_path_label.setText("Data YAML: Not generated yet.")
            
            # Automatically start preparation after selection
            if hasattr(self, 'start_data_preparation'):
                self.start_data_preparation()
            else:
                self.log("Error: start_data_preparation method not found.")
        else:
            self.log("Dataset selection cancelled.")
######################################################################################################################

    def create_page_widget(self):
        """Helper to create a standard page widget with scroll area."""
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(15,15,15,15) # Padding for page content
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("pageScrollArea")
        scroll_area.setFrameShape(QFrame.NoFrame) # No border for scroll area itself

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0,0,0,0)
        content_layout.setSpacing(15) # Spacing between group boxes
        
        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)
        
        return page, content_layout # Return the layout where group boxes will be added

    # --- Page Creation Methods ---
    def create_home_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("HomePage")

        title = QLabel("Welcome to VisionCraft Studio")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        intro_text = QLabel(
            "Your integrated environment for YOLOv8 model development.\n\n"
            "Navigate using the tabs below:\n"
            "  •  <b>Data:</b> Prepare and manage your datasets.\n"
            "  •  <b>Train:</b> Configure and run model training.\n"
            "  •  <b>Deploy:</b> Test models, run live inference, and export.\n"
            "  •  <b>Settings:</b> Manage configurations and dependencies."
        )
        intro_text.setWordWrap(True)
        intro_text.setObjectName("introText")
        layout.addWidget(intro_text)
        
        # Quick Stats (Example)
        stats_group = QGroupBox("Quick Overview")
        stats_layout = QVBoxLayout()
        self.models_trained_label = QLabel("Models Trained: 0 (Placeholder)") # You can update this
        self.datasets_prepared_label = QLabel("Datasets Prepared: 0 (Placeholder)")
        stats_layout.addWidget(self.models_trained_label)
        stats_layout.addWidget(self.datasets_prepared_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return page

    def create_data_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("DataPage")

        group = QGroupBox("Dataset Preparation")
        group.setObjectName("contentGroup")
        data_layout = QVBoxLayout()

        self.data_zip_label = QLabel("No dataset (.zip) selected.")
        self.data_zip_label.setWordWrap(True)
        load_button = QPushButton(QIcon("path/to/folder_open_icon.png"), " Load Custom Dataset")
        load_button.clicked.connect(self.select_dataset_zip)
        data_layout.addWidget(self.data_zip_label)
        data_layout.addWidget(load_button)

        self.data_prep_progress = QProgressBar()
        self.data_prep_status_label = QLabel("Status: Waiting for dataset...")
        data_layout.addWidget(self.data_prep_progress)
        data_layout.addWidget(self.data_prep_status_label)
        
        # Add a label to show the path to the generated data.yaml
        self.data_yaml_path_label = QLabel("Data YAML: Not generated yet.")
        self.data_yaml_path_label.setWordWrap(True)
        data_layout.addWidget(self.data_yaml_path_label)

        group.setLayout(data_layout)
        layout.addWidget(group)
        layout.addStretch()
        return page

    def create_train_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("TrainPage")

        # Model Selection Group
        model_group = QGroupBox("1. Model Selection for Training")
        model_group.setObjectName("contentGroup")
        model_layout = QVBoxLayout()
        
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Base YOLO Model:"))
        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(get_yolo_models())
        self.train_model_combo.setCurrentText("yolov8s.pt")
        combo_layout.addWidget(self.train_model_combo)
        model_layout.addLayout(combo_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Training Parameters Group
        params_group = QGroupBox("2. Training Parameters")
        params_group.setObjectName("contentGroup")
        params_form_layout = QVBoxLayout() # Using QVBoxLayout for better spacing

        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(25) # Reduced default for quicker tests
        epochs_layout.addWidget(self.epochs_spinbox)
        params_form_layout.addLayout(epochs_layout)

        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel("Image Size (px):"))
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 2048)
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setSingleStep(32)
        imgsz_layout.addWidget(self.imgsz_spinbox)
        params_form_layout.addLayout(imgsz_layout)
        
        # Project Name for Runs
        project_name_layout = QHBoxLayout()
        project_name_layout.addWidget(QLabel("Training Run Project Name:"))
        self.train_project_name_input = QComboBox() # Use QLineEdit if free text is better
        self.train_project_name_input.setEditable(True)
        self.train_project_name_input.addItems(["yolo_gui_runs", "my_object_detection_project"])
        self.train_project_name_input.setCurrentText("yolo_gui_runs")
        project_name_layout.addWidget(self.train_project_name_input)
        params_form_layout.addLayout(project_name_layout)


        params_group.setLayout(params_form_layout)
        layout.addWidget(params_group)

        # Action Group
        action_group = QGroupBox("3. Start Training")
        action_group.setObjectName("contentGroup")
        action_layout = QVBoxLayout()
        self.train_button = QPushButton(QIcon("path/to/play_icon.png"), " Start Training")
        self.train_button.clicked.connect(self.start_training_process)
        self.train_button.setEnabled(False) # Enable after data prep
        action_layout.addWidget(self.train_button)
        self.train_status_label = QLabel("Status: Configure data and model first.")
        action_layout.addWidget(self.train_status_label)
        self.trained_model_path_display_label = QLabel("Trained Model: Not yet trained.")
        self.trained_model_path_display_label.setWordWrap(True)
        action_layout.addWidget(self.trained_model_path_display_label)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        layout.addStretch()
        return page

    def create_deploy_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("DeployPage")

        # --- Load Model for Inference ---
        load_model_group = QGroupBox("1. Load Model for Inference/Export")
        load_model_group.setObjectName("contentGroup")
        load_model_layout = QVBoxLayout()
        self.inference_model_status_label = QLabel("No model loaded for deployment.")
        self.inference_model_status_label.setWordWrap(True)
        self.load_trained_model_button = QPushButton(QIcon("path/to/model_icon.png"), " Load Trained Model (.pt)")
        self.load_trained_model_button.clicked.connect(self.select_model_for_inference)
        load_model_layout.addWidget(self.inference_model_status_label)
        load_model_layout.addWidget(self.load_trained_model_button)
        load_model_group.setLayout(load_model_layout)
        layout.addWidget(load_model_group)

        # --- Live Inference Part ---
        inference_group = QGroupBox("2. Live Inference / Test")
        inference_group.setObjectName("contentGroup")
        inf_layout = QVBoxLayout()

        self.video_feed_label = QLabel()
        self.video_feed_label.setMinimumSize(640, 480)
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_feed_label.setStyleSheet("border: 1px solid #CCCCCC; background-color: #333333;")
        self.video_feed_label.setText("Video Feed")
        inf_layout.addWidget(self.video_feed_label)
        
        # FPS and Object Count Display
        stats_display_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("infoLabel")
        self.detected_objects_label = QLabel("Objects: --")
        self.detected_objects_label.setObjectName("infoLabel")
        stats_display_layout.addWidget(self.fps_label)
        stats_display_layout.addStretch()
        stats_display_layout.addWidget(self.detected_objects_label)
        inf_layout.addLayout(stats_display_layout)


        inf_controls_layout = QGridLayout() # Use QGridLayout for better alignment
        self.webcam_combo = QComboBox()
        self.update_webcam_list() # Populate webcam list
        
        self.start_webcam_button = QPushButton(QIcon("path/to/webcam_icon.png"), " Start Webcam")
        self.start_webcam_button.clicked.connect(self.run_webcam_inference)
        
        self.load_media_button = QPushButton(QIcon("path/to/file_video_icon.png"), " Load Video/Image File")
        self.load_media_button.clicked.connect(self.run_file_inference)
        
        self.stop_inference_button = QPushButton(QIcon("path/to/stop_icon.png"), " Stop Inference")
        self.stop_inference_button.clicked.connect(self.stop_live_inference)
        
        inf_controls_layout.addWidget(QLabel("Source:"), 0, 0)
        inf_controls_layout.addWidget(self.webcam_combo, 0, 1)
        inf_controls_layout.addWidget(self.start_webcam_button, 0, 2)
        inf_controls_layout.addWidget(self.load_media_button, 1, 1, 1, 2) # Span 2 columns
        inf_controls_layout.addWidget(self.stop_inference_button, 2, 1, 1, 2) # Span 2 columns
        
        inf_layout.addLayout(inf_controls_layout)
        inference_group.setLayout(inf_layout)
        layout.addWidget(inference_group)

        # --- Export Part ---
        export_group = QGroupBox("3. Model Conversion (Export)")
        export_group.setObjectName("contentGroup")
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Export to Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(get_export_formats())
        self.export_model_button = QPushButton(QIcon("path/to/export_icon.png"), " Convert & Export Model")
        self.export_model_button.clicked.connect(self.run_export_model)
        
        exp_layout.addWidget(self.export_format_combo)
        exp_layout.addWidget(self.export_model_button)
        export_group.setLayout(exp_layout)
        layout.addWidget(export_group)
        self.export_status_label = QLabel("Export Status: Ready")
        layout.addWidget(self.export_status_label)


        layout.addStretch()
        self.update_inference_controls_state() # Initial state
        return page

    def create_settings_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("SettingsPage")

        # Ultralytics Installation
        ultralytics_group = QGroupBox("Ultralytics Dependency Management")
        ultralytics_group.setObjectName("contentGroup")
        ult_layout = QVBoxLayout()
        self.ultralytics_status_label = QLabel("Ultralytics Status: Unknown")
        self.install_ultralytics_button = QPushButton(QIcon("path/to/install_icon.png"), " Install/Verify Ultralytics")
        self.install_ultralytics_button.clicked.connect(self.manage_ultralytics_installation)
        ult_layout.addWidget(self.ultralytics_status_label)
        ult_layout.addWidget(self.install_ultralytics_button)
        ultralytics_group.setLayout(ult_layout)
        layout.addWidget(ultralytics_group)

        # Compute Device Selection
        device_group = QGroupBox("Compute Device Configuration")
        device_group.setObjectName("contentGroup")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Select Default Compute Device:"))
        self.device_combo = QComboBox()
        self.populate_device_combo()
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Theme Selection (Example of an added feature)
        theme_group = QGroupBox("Appearance")
        theme_group.setObjectName("contentGroup")
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("UI Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark (Experimental)"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)


        layout.addStretch()
        return page

    # --- Styling ---
    def apply_stylesheet(self, theme="light"):
        self.current_theme = theme
        # Basic clean theme. For a true Apple feel, this needs to be much more detailed.
        # Consider using existing QSS themes or building one with more specific selectors.
        
        # Common font
        font = QFont("SF Pro Display", 10) # macOS Font, use "Segoe UI" for Windows, "Cantarell" for Linux
        if sys.platform == "win32":
            font.setFamily("Segoe UI")
        elif sys.platform == "darwin":
             font.setFamily("SF Pro Display") # Or Helvetica Neue
        else: # Linux
            font.setFamily("Cantarell") # Or Noto Sans
        QApplication.setFont(font)


        light_palette = {
            "window_bg": "#ECECEC", # Off-white
            "nav_bar_bg": "#F8F8F8",
            "nav_button_bg": "transparent",
            "nav_button_hover_bg": "#E0E0E0",
            "nav_button_checked_bg": "#D0D0D0",
            "nav_button_text": "#333333",
            "text_color": "#222222",
            "label_header_color": "#007AFF", # Apple Blue
            "border_color": "#CCCCCC",
            "group_bg": "#FFFFFF", # White for group boxes
            "button_bg": "#F0F0F0",
            "button_hover_bg": "#E0E0E0",
            "button_pressed_bg": "#D0D0D0",
            "button_text": "#333333",
            "input_bg": "#FFFFFF",
            "input_border": "#BDBDBD",
            "disabled_text": "#AAAAAA",
            "disabled_bg": "#E0E0E0",
            "scroll_bar_bg": "#F0F0F0",
            "scroll_bar_handle": "#C0C0C0",
            "log_bg": "#FDFDFD",
            "log_text": "#333333",
        }

        dark_palette = { # Experimental dark theme
            "window_bg": "#2D2D2D",
            "nav_bar_bg": "#1E1E1E",
            "nav_button_bg": "transparent",
            "nav_button_hover_bg": "#4A4A4A",
            "nav_button_checked_bg": "#5A5A5A",
            "nav_button_text": "#E0E0E0",
            "text_color": "#F0F0F0",
            "label_header_color": "#0A84FF", # Lighter Apple Blue for dark mode
            "border_color": "#454545",
            "group_bg": "#3C3C3C",
            "button_bg": "#505050",
            "button_hover_bg": "#606060",
            "button_pressed_bg": "#707070",
            "button_text": "#F0F0F0",
            "input_bg": "#454545",
            "input_border": "#606060",
            "disabled_text": "#777777",
            "disabled_bg": "#404040",
            "scroll_bar_bg": "#3A3A3A",
            "scroll_bar_handle": "#606060",
            "log_bg": "#252525",
            "log_text": "#D0D0D0",
        }
        
        p = light_palette if theme == "light" else dark_palette

        qss = f"""
            QMainWindow, QWidget#central_widget {{
                background-color: {p["window_bg"]};
            }}
            QWidget#HomePage, QWidget#DataPage, QWidget#TrainPage, QWidget#DeployPage, QWidget#SettingsPage {{
                background-color: {p["window_bg"]}; /* Ensure pages also get background */
            }}
            QScrollArea#pageScrollArea {{
                background-color: transparent; /* Make scroll area transparent */
                border: none;
            }}
            QScrollArea#pageScrollArea > QWidget > QWidget {{ /* Target the content_widget inside scrollarea */
                background-color: transparent;
            }}
            QWidget#nav_widget {{ /* Navigation bar specific styling */
                background-color: {p["nav_bar_bg"]};
                border-top: 1px solid {p["border_color"]};
            }}
            QPushButton.navButton {{
                background-color: {p["nav_button_bg"]};
                color: {p["nav_button_text"]};
                border: none; /* No individual borders for nav buttons */
                padding: 10px;
                font-size: 10pt; /* Slightly larger font for nav */
                font-weight: bold;
            }}
            QPushButton.navButton:hover {{
                background-color: {p["nav_button_hover_bg"]};
            }}
            QPushButton.navButton:checked {{
                background-color: {p["nav_button_checked_bg"]};
                border-bottom: 2px solid {p["label_header_color"]}; /* Indicator for active tab */
            }}
            QGroupBox {{
                background-color: {p["group_bg"]};
                border: 1px solid {p["border_color"]};
                border-radius: 8px; /* Rounded corners for groups */
                margin-top: 10px;
                padding: 10px;
                 font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px 5px 5px; /* Adjust title padding */
                color: {p["label_header_color"]};
                 font-size: 11pt;
            }}
            QLabel, QCheckBox {{
                color: {p["text_color"]};
                font-size: 9pt;
            }}
            QLabel#pageTitle {{
                font-size: 18pt;
                font-weight: bold;
                color: {p["label_header_color"]};
                padding-bottom: 10px;
            }}
             QLabel#introText {{
                font-size: 10pt;
                line-height: 1.5; /* For better readability */
            }}
            QLabel#infoLabel {{ /* For FPS, Object Count */
                font-size: 10pt;
                font-weight: bold;
                color: {p["label_header_color"]};
            }}
            QPushButton {{
                background-color: {p["button_bg"]};
                color: {p["button_text"]};
                border: 1px solid {p["border_color"]};
                padding: 8px 15px;
                border-radius: 5px;
                font-size: 9pt;
            }}
            QPushButton:hover {{
                background-color: {p["button_hover_bg"]};
            }}
            QPushButton:pressed {{
                background-color: {p["button_pressed_bg"]};
            }}
            QPushButton:disabled {{
                background-color: {p["disabled_bg"]};
                color: {p["disabled_text"]};
            }}
            QComboBox, QSpinBox, QLineEdit {{
                background-color: {p["input_bg"]};
                color: {p["text_color"]};
                border: 1px solid {p["input_border"]};
                padding: 5px;
                border-radius: 4px;
                min-height: 20px; /* Ensure consistent height */
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{ /* Dropdown list style */
                background-color: {p["input_bg"]};
                color: {p["text_color"]};
                border: 1px solid {p["input_border"]};
                selection-background-color: {p["label_header_color"]};
            }}
            QProgressBar {{
                border: 1px solid {p["border_color"]};
                border-radius: 4px;
                text-align: center;
                color: {p["text_color"]};
            }}
            QProgressBar::chunk {{
                background-color: {p["label_header_color"]};
                border-radius: 3px;
            }}
            QTextEdit {{
                background-color: {p["log_bg"]};
                color: {p["log_text"]};
                border: 1px solid {p["border_color"]};
                border-radius: 4px;
                font-family: "Monaco", "Consolas", "Courier New", monospace; /* Monospaced font for logs */
            }}
             QScrollBar:vertical {{
                 border: none;
                 background: {p["scroll_bar_bg"]};
                 width: 10px;
                 margin: 0px 0px 0px 0px;
             }}
             QScrollBar::handle:vertical {{
                 background: {p["scroll_bar_handle"]};
                 min-height: 20px;
                 border-radius: 5px;
             }}
             QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                 border: none;
                 background: none;
             }}
             QScrollBar:horizontal {{
                 border: none;
                 background: {p["scroll_bar_bg"]};
                 height: 10px;
                 margin: 0px 0px 0px 0px;
             }}
             QScrollBar::handle:horizontal {{
                 background: {p["scroll_bar_handle"]};
                 min-width: 20px;
                 border-radius: 5px;
             }}
             QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                 border: none;
                 background: none;
             }}
        """
        self.setStyleSheet(qss)

    def change_theme(self, theme_name):
        self.log(f"Changing theme to {theme_name.lower().split(' ')[0]}")
        self.apply_stylesheet(theme_name.lower().split(' ')[0])
        
    # --- Logging ---
    def log(self, message):
        """Appends a message to the log area."""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum()) # Auto-scroll
        print(message) # Also print to console

    # --- UI Update/Helper Functions ---
    def update_ultralytics_status_display(self):
        if ULTRALYTICS_INSTALLED:
            self.ultralytics_status_label.setText("Ultralytics Status: Installed ✔️")
            self.install_ultralytics_button.setIcon(QIcon("path/to/check_icon.png"))  # Set icon
            self.install_ultralytics_button.setText(" Verify/Reinstall Ultralytics") # Set text
            self.install_ultralytics_button.setEnabled(True)
        else:
            self.ultralytics_status_label.setText("Ultralytics Status: Not Installed ❌")
            self.install_ultralytics_button.setIcon(QIcon("path/to/install_icon.png"))  # Set icon
            self.install_ultralytics_button.setText(" Install Ultralytics")          # Set text
            self.install_ultralytics_button.setEnabled(True)
        
        # Enable/disable features based on Ultralytics status
        # Ensure self.train_button exists before trying to setEnabled on it, especially if UI creation order matters
        if hasattr(self, 'train_button') and self.train_button:
            self.train_button.setEnabled(ULTRALYTICS_INSTALLED and self.data_yaml_path is not None)
        
        if hasattr(self, 'update_inference_controls_state'): # Ensure this method itself is safe to call
            self.update_inference_controls_state()


    def populate_device_combo(self):
        self.device_combo.clear()
        self.device_combo.addItem("cpu")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.device_combo.addItem(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
            self.device_combo.setCurrentIndex(1 if torch.cuda.device_count() > 0 else 0)
        else:
            self.device_combo.setCurrentIndex(0)

    def get_selected_device(self):
        device_text = self.device_combo.currentText()
        if device_text == 'cpu':
            return 'cpu'
        elif 'cuda' in device_text:
            return device_text.split(' ')[0] # e.g. "cuda:0"
        return 'cpu' # Default fallback

    def update_webcam_list(self):
        self.webcam_combo.clear()
        cams = get_available_cameras()
        if cams:
            self.webcam_combo.addItems([f"Webcam {i}" for i in cams])
        else:
            self.webcam_combo.addItem("No webcams found")
            self.webcam_combo.setEnabled(False)


    def update_inference_controls_state(self):
        model_loaded = bool(self.trained_model_path_for_inference)
        ultralytics_ready = ULTRALYTICS_INSTALLED

        can_start_inference = model_loaded and ultralytics_ready
        
        self.start_webcam_button.setEnabled(can_start_inference and self.webcam_combo.count() > 0 and self.webcam_combo.currentText() != "No webcams found")
        self.load_media_button.setEnabled(can_start_inference)
        self.export_model_button.setEnabled(model_loaded and ultralytics_ready)
        
        # Stop button should only be enabled if inference is running
        # is_inf_running = self.inference_thread and self.inference_thread.isRunning()
        is_inf_running = self.inference_thread is not None and self.inference_thread.isRunning()
        self.stop_inference_button.setEnabled(is_inf_running)

        if not model_loaded:
            self.inference_model_status_label.setText("No model loaded for deployment. Please load a .pt model.")
            self.export_status_label.setText("Export Status: Load a model first.")
        if not ultralytics_ready:
             self.inference_model_status_label.setText("Ultralytics not installed. Please install from Settings.")
             self.export_status_label.setText("Export Status: Ultralytics not installed.")


    # --- Action Slots ---

    # Settings Page Actions
    def manage_ultralytics_installation(self):
        self.install_ultralytics_button.setEnabled(False)
        self.log("Starting Ultralytics installation/verification process...")
        self.install_thread = InstallThread()
        self.install_thread.log_message.connect(self.log)
        self.install_thread.finished.connect(self.on_ultralytics_install_finished)
        self.install_thread.start()

    def on_ultralytics_install_finished(self, success, message):
        self.log(message)
        self.update_ultralytics_status_display()
        # Re-populate device combo as CUDA might become available after install
        self.populate_device_combo()


    # Data Page Actions
    def select_dataset_zip(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Dataset Zip File", "", "Zip Files (*.zip)", options=options)
        if fileName:
            self.dataset_zip_path = fileName
            self.data_zip_label.setText(f"Selected: {os.path.basename(fileName)}")
            self.log(f"Dataset ZIP selected: {fileName}")
            self.data_yaml_path = None # Reset previous YAML path
            self.data_yaml_path_label.setText("Data YAML: Not generated yet.")
            self.start_data_preparation()

    def start_data_preparation(self):
        if not self.dataset_zip_path:
            self.log("Error: No dataset ZIP file selected.")
            self.data_prep_status_label.setText("Status: Select a dataset first ❌")
            return

        self.log("Starting data preparation thread...")
        self.data_prep_status_label.setText("Status: Preparing dataset...")
        self.data_prep_progress.setValue(0)
        
        # Use the central work_dir for data processing
        data_processing_subdir = os.path.join(self.work_dir, "dataset_processing")
        os.makedirs(data_processing_subdir, exist_ok=True)

        self.data_prep_thread = DataPrepThread(self.dataset_zip_path, data_processing_subdir)
        self.data_prep_thread.log_message.connect(self.log)
        self.data_prep_thread.progress.connect(self.update_data_prep_display)
        self.data_prep_thread.finished.connect(self.on_data_prep_complete)
        self.data_prep_thread.start()

    def update_data_prep_display(self, message, value):
        self.data_prep_status_label.setText(f"Status: {message}")
        self.data_prep_progress.setValue(value)

    def on_data_prep_complete(self, success, path_or_message):
        if success:
            self.data_yaml_path = path_or_message
            self.log(f"Data preparation successful. YAML config: {self.data_yaml_path}")
            self.data_prep_status_label.setText("Status: Data Ready ✔️")
            self.data_yaml_path_label.setText(f"Data YAML: {self.data_yaml_path}")
            self.train_button.setEnabled(ULTRALYTICS_INSTALLED)
        else:
            self.log(f"Data preparation failed: {path_or_message}")
            self.data_prep_status_label.setText(f"Status: Failed ❌ ({path_or_message})")
            self.data_yaml_path_label.setText("Data YAML: Generation failed.")
            self.train_button.setEnabled(False)
        self.data_prep_progress.setValue(100 if success else 0)


    # Train Page Actions
    def start_training_process(self):
        if not self.data_yaml_path:
            self.log("Error: Dataset not prepared or YAML path not set.")
            self.train_status_label.setText("Status: Prepare dataset first ❌")
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics is not installed. Cannot start training.")
            self.train_status_label.setText("Status: Install Ultralytics from Settings ❌")
            return

        self.train_button.setEnabled(False)
        self.train_status_label.setText("Status: Starting training...")
        self.log("Initiating training...")

        model_to_train = self.train_model_combo.currentText()
        epochs = self.epochs_spinbox.value()
        imgsz = self.imgsz_spinbox.value()
        device_to_use = self.get_selected_device()
        project_name = self.train_project_name_input.currentText()
        if not project_name.strip():
            project_name = "yolo_gui_runs" # Default if empty
            self.log("Project name empty, defaulting to 'yolo_gui_runs'.")


        self.train_thread = TrainThread(model_to_train, self.data_yaml_path, epochs, imgsz, device_to_use, project_name)
        self.train_thread.progress_update.connect(self.log) # Log detailed progress from thread
        self.train_thread.finished.connect(self.on_training_session_finished)
        self.train_thread.start()

    def on_training_session_finished(self, success, model_path_or_message):
        self.train_button.setEnabled(True)
        if success:
            # self.trained_model_path_for_inference = model_path_or_message # Optionally auto-load for deploy
            self.log(f"Training successful! Model saved at: {model_path_or_message}")
            self.train_status_label.setText(f"Status: Training Complete ✔️")
            self.trained_model_path_display_label.setText(f"Latest Trained Model: {model_path_or_message}")
            # self.update_inference_controls_state() # If auto-loading
        else:
            self.log(f"Training failed: {model_path_or_message}")
            self.train_status_label.setText(f"Status: Training Failed ❌")
            self.trained_model_path_display_label.setText("Trained Model: Error during training.")


    # Deploy Page Actions
    def select_model_for_inference(self, model_path=None):
        if not model_path or not isinstance(model_path, str): # Allow direct path pass for auto-load
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            fileName, _ = QFileDialog.getOpenFileName(self, "Select Trained Model File (.pt)", "", "PyTorch Models (*.pt)", options=options)
            if not fileName:
                return
            model_path = fileName
            
        self.trained_model_path_for_inference = model_path
        self.log(f"Model selected for inference/export: {self.trained_model_path_for_inference}")
        self.inference_model_status_label.setText(f"Active Model: {os.path.basename(self.trained_model_path_for_inference)}")
        self.update_inference_controls_state()
        # Clear previous inference artifacts
        self.video_feed_label.setText("Video Feed (Model Loaded)")
        self.fps_label.setText("FPS: --")
        self.detected_objects_label.setText("Objects: --")


    def run_live_inference(self, source_type, source_path):
        if not self.trained_model_path_for_inference:
            self.log("Error: No trained model loaded for inference.")
            self.video_feed_label.setText("Error: No model loaded!")
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics not installed.")
            self.video_feed_label.setText("Error: Ultralytics not installed!")
            return
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Inference is already running. Stop it first.")
            return

        self.log(f"Starting live inference on {source_type}: {source_path}")
        self.video_feed_label.setText("Starting inference...")
        device_to_use = self.get_selected_device()

        self.inference_thread = InferenceThread(self.trained_model_path_for_inference, source_type, source_path, device_to_use)
        self.inference_thread.log_message.connect(self.log)
        self.inference_thread.frame_ready.connect(self.update_inference_video_feed)
        self.inference_thread.finished.connect(self.on_live_inference_finished)
        
        self.start_webcam_button.setEnabled(False)
        self.load_media_button.setEnabled(False)
        self.stop_inference_button.setEnabled(True)
        self.inference_thread.start()


    def run_webcam_inference(self):
        cam_text = self.webcam_combo.currentText() # "Webcam 0"
        if "No webcams" in cam_text:
            self.log("No webcam selected or available.")
            return
        cam_index_str = cam_text.split()[-1]
        self.run_live_inference('webcam', cam_index_str)

    def run_file_inference(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video or Image File", "",
                                                  "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.webp);;Video Files (*.mp4 *.avi *.mov *.mkv);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp)", 
                                                  options=options)
        if fileName:
            if fileName.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                self.run_live_inference('image', fileName)
            else:
                self.run_live_inference('video', fileName)

    def stop_live_inference(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Sending stop signal to inference process...")
            self.inference_thread.stop()
            # Buttons will be re-enabled in on_live_inference_finished
            self.stop_inference_button.setEnabled(False) # Disable immediately
        else:
            self.log("No active inference process to stop.")


    def update_inference_video_feed(self, q_image, fps, object_count):
        self.video_feed_label.setPixmap(QPixmap.fromImage(q_image))
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.detected_objects_label.setText(f"Objects: {object_count}")

    def on_live_inference_finished(self, message):
        self.log(f"Inference process ended: {message}")
        self.video_feed_label.setText(f"Inference Stopped/Finished\n({message})")
        self.fps_label.setText("FPS: --")
        self.detected_objects_label.setText("Objects: --")
        
        self.inference_thread = None # Clear the thread instance
        self.update_inference_controls_state() # Reset button states


    def run_export_model(self):
        if not self.trained_model_path_for_inference:
            self.log("Error: No trained model loaded to export.")
            self.export_status_label.setText("Export Status: Load a .pt model first ❌")
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics is not installed for export.")
            self.export_status_label.setText("Export Status: Ultralytics not installed ❌")
            return

        export_format_selected = self.export_format_combo.currentText()
        self.log(f"Preparing to export model to {export_format_selected} format...")
        self.export_status_label.setText(f"Status: Exporting to {export_format_selected}...")
        self.export_model_button.setEnabled(False)

        self.export_thread = ExportThread(self.trained_model_path_for_inference, export_format_selected)
        self.export_thread.log_message.connect(self.log)
        self.export_thread.finished.connect(self.on_export_model_finished)
        self.export_thread.start()

    def on_export_model_finished(self, success, path_or_message):
        self.log(path_or_message)
        if success:
            self.export_status_label.setText(f"Status: Export Successful ✔️ ({os.path.basename(path_or_message)})")
        else:
            self.export_status_label.setText(f"Status: Export Failed ❌")
        self.export_model_button.setEnabled(True) # Re-enable button


    def closeEvent(self, event):
        """Ensure threads are stopped when closing."""
        self.log("Application closing. Stopping active threads...")
        self.stop_live_inference() # Will join if running

        # Wait for threads to finish if necessary, though Qt usually handles this
        # if self.inference_thread and self.inference_thread.isRunning():
        #     self.inference_thread.wait()
        # Add for other long-running threads if they aren't managed by immediate stop signals

        # Clean up work directory if desired (optional)
        # if os.path.exists(self.work_dir):
        #     try:
        #         shutil.rmtree(self.work_dir)
        #         self.log(f"Cleaned up work directory: {self.work_dir}")
        #     except Exception as e:
        #         self.log(f"Could not remove work directory {self.work_dir}: {e}")

        self.log("Exiting VisionCraft Studio.")
        event.accept()

# --- Run the App ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Forcing a style can sometimes help with consistency across platforms
    # app.setStyle("Fusion") # or "Windows", "Motif", "GTK+"
    
    window = ModernYoloGUI()
    window.show()
    sys.exit(app.exec_())