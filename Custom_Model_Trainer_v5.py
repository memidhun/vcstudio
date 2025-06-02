import sys
import os
import subprocess
import zipfile
import shutil
import yaml
import cv2
import torch
import numpy as np # For dummy frame in dependency check
import random
import glob
import time
import webbrowser # For opening links

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLabel, QSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QScrollArea, QSizePolicy,
    QStackedWidget, QFrame, QGridLayout, QCheckBox, QMessageBox,
    QSpacerItem, QStatusBar, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QUrl
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QDesktopServices

# Try to import ultralytics, prompt for installation if not found
try:
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_requirements # For specific checks if needed
    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False

# --- Constants for Theming and Icons ---
THEME_LIGHT = "light"
THEME_DARK = "dark"
ICON_SIZE = QSize(24, 24) # Default icon size for navigation

# --- Dracula Theme Colors (VS Code Inspired) ---
DRACULA_BACKGROUND = "#282a36"
DRACULA_CURRENT_LINE = "#44475a"
DRACULA_FOREGROUND = "#f8f8f2"
DRACULA_COMMENT = "#6272a4"
DRACULA_CYAN = "#8be9fd"
DRACULA_GREEN = "#50fa7b"
DRACULA_ORANGE = "#ffb86c"
DRACULA_PINK = "#ff79c6"
DRACULA_PURPLE = "#bd93f9"
DRACULA_RED = "#ff5555"
DRACULA_YELLOW = "#f1fa8c"
DRACULA_SCROLLBAR_BG = "#313341"
DRACULA_SCROLLBAR_HANDLE = "#44475a"
DRACULA_BUTTON_BG = "#44475a"
DRACULA_BUTTON_HOVER_BG = "#5a5c72"
DRACULA_BUTTON_PRESSED_BG = "#3b3d4f"
DRACULA_INPUT_BG = "#3b3d4f"
DRACULA_INPUT_BORDER = "#6272a4"
DRACULA_GROUPBOX_BORDER = "#6272a4"
DRACULA_TEXT_EDIT_BG = "#21222c"
DRACULA_SELECTED_ITEM_BG = "#44475a"

DARK_STYLESHEET = f"""
    QMainWindow {{
        background-color: {DRACULA_BACKGROUND};
        color: {DRACULA_FOREGROUND};
    }}
    QWidget {{
        background-color: {DRACULA_BACKGROUND};
        color: {DRACULA_FOREGROUND};
        border-radius: 5px; /* Rounded corners for widgets */
    }}
    QGroupBox {{
        background-color: {DRACULA_BACKGROUND};
        border: 1px solid {DRACULA_GROUPBOX_BORDER};
        border-radius: 8px;
        margin-top: 10px; /* Space for title */
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left; /* Position at top left */
        padding: 0 5px 0 5px;
        color: {DRACULA_PURPLE};
        font-weight: bold;
    }}
    QLabel {{
        color: {DRACULA_FOREGROUND};
        background-color: transparent; /* Ensure labels don't have their own opaque background */
        padding: 2px;
    }}
    QPushButton {{
        background-color: {DRACULA_BUTTON_BG};
        color: {DRACULA_FOREGROUND};
        border: 1px solid {DRACULA_COMMENT};
        padding: 8px 12px;
        border-radius: 5px;
        min-height: 20px; /* Ensure buttons have a decent height */
    }}
    QPushButton:hover {{
        background-color: {DRACULA_BUTTON_HOVER_BG};
    }}
    QPushButton:pressed {{
        background-color: {DRACULA_BUTTON_PRESSED_BG};
    }}
    QPushButton:disabled {{
        background-color: {DRACULA_CURRENT_LINE};
        color: {DRACULA_COMMENT};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {DRACULA_INPUT_BG};
        color: {DRACULA_FOREGROUND};
        border: 1px solid {DRACULA_INPUT_BORDER};
        padding: 5px;
        border-radius: 5px;
        min-height: 20px;
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox::down-arrow {{
        image: url(icons/dark/arrow_down.png); /* Placeholder for a themed dropdown arrow */
        width: 12px;
        height: 12px;
    }}
    QComboBox QAbstractItemView {{ /* Dropdown list style */
        background-color: {DRACULA_INPUT_BG};
        border: 1px solid {DRACULA_INPUT_BORDER};
        selection-background-color: {DRACULA_SELECTED_ITEM_BG};
        color: {DRACULA_FOREGROUND};
    }}
    QTextEdit {{
        background-color: {DRACULA_TEXT_EDIT_BG};
        color: {DRACULA_FOREGROUND};
        border: 1px solid {DRACULA_INPUT_BORDER};
        border-radius: 5px;
        padding: 5px;
    }}
    QProgressBar {{
        border: 1px solid {DRACULA_COMMENT};
        border-radius: 5px;
        text-align: center;
        color: {DRACULA_FOREGROUND};
        background-color: {DRACULA_INPUT_BG};
    }}
    QProgressBar::chunk {{
        background-color: {DRACULA_GREEN};
        border-radius: 4px; /* Slightly smaller radius than the bar itself */
        margin: 1px; /* Small margin for the chunk */
    }}
    QScrollBar:vertical {{
        border: none;
        background: {DRACULA_SCROLLBAR_BG};
        width: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {DRACULA_SCROLLBAR_HANDLE};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}
    QScrollBar:horizontal {{
        border: none;
        background: {DRACULA_SCROLLBAR_BG};
        height: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {DRACULA_SCROLLBAR_HANDLE};
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        border: none;
        background: none;
    }}
    QFrame#video_display_label {{ /* Specific styling for video display */
        border: 2px solid {DRACULA_COMMENT};
        background-color: {DRACULA_TEXT_EDIT_BG}; /* Darker background for video area */
    }}
    /* Styling for the navigation buttons */
    QPushButton.navButton {{
        background-color: transparent;
        border: none;
        color: {DRACULA_FOREGROUND};
        padding: 10px;
        text-align: left; /* Align text and icon to the left */
    }}
    QPushButton.navButton:hover {{
        background-color: {DRACULA_CURRENT_LINE}; /* Highlight on hover */
    }}
    QPushButton.navButton:checked {{ /* Style for the active/selected button */
        background-color: {DRACULA_PURPLE};
        color: {DRACULA_BACKGROUND}; /* Contrasting text for selected */
        font-weight: bold;
    }}
    QLabel#clickableLink {{
        color: {DRACULA_CYAN};
        text-decoration: underline;
    }}
    QLabel#clickableLink:hover {{
        color: {DRACULA_PINK};
    }}
    QStatusBar {{
        background-color: {DRACULA_BACKGROUND};
        color: {DRACULA_COMMENT};
    }}
    QStatusBar::item {{
        border: none; /* Remove border from status bar items */
    }}
"""

LIGHT_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #f0f0f0; /* Light gray background */
        color: #333; /* Dark text */
    }
    QGroupBox {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        border-radius: 8px;
        margin-top: 10px;
        padding: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px 0 5px;
        color: #555599; /* A muted purple/blue */
        font-weight: bold;
    }
    QLabel {
        color: #333;
        background-color: transparent;
        padding: 2px;
    }
    QPushButton {
        background-color: #e0e0e0; /* Light button background */
        color: #333;
        border: 1px solid #b0b0b0;
        padding: 8px 12px;
        border-radius: 5px;
        min-height: 20px;
    }
    QPushButton:hover {
        background-color: #d0d0d0; /* Slightly darker on hover */
    }
    QPushButton:pressed {
        background-color: #c0c0c0; /* Even darker when pressed */
    }
    QPushButton:disabled {
        background-color: #dcdcdc;
        color: #a0a0a0;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #ffffff; /* White input fields */
        color: #333;
        border: 1px solid #b0b0b0;
        padding: 5px;
        border-radius: 5px;
        min-height: 20px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox::down-arrow {
        image: url(icons/light/arrow_down.png); /* Placeholder */
        width: 12px;
        height: 12px;
    }
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        border: 1px solid #b0b0b0;
        selection-background-color: #a0a0d0; /* A light purple for selection */
        color: #333;
    }
    QTextEdit {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #b0b0b0;
        border-radius: 5px;
        padding: 5px;
    }
    QProgressBar {
        border: 1px solid #b0b0b0;
        border-radius: 5px;
        text-align: center;
        color: #333;
        background-color: #e0e0e0;
    }
    QProgressBar::chunk {
        background-color: #50c878; /* A nice green */
        border-radius: 4px;
        margin: 1px;
    }
    QScrollBar:vertical {
        border: none;
        background: #e0e0e0;
        width: 10px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:vertical {
        background: #c0c0c0;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar:horizontal {
        border: none;
        background: #e0e0e0;
        height: 10px;
        margin: 0px 0px 0px 0px;
    }
    QScrollBar::handle:horizontal {
        background: #c0c0c0;
        min-width: 20px;
        border-radius: 5px;
    }
    QFrame#video_display_label {
        border: 2px solid #b0b0b0;
        background-color: #ffffff;
    }
    QPushButton.navButton {
        background-color: transparent;
        border: none;
        color: #333;
        padding: 10px;
        text-align: left;
    }
    QPushButton.navButton:hover {
        background-color: #d0d0d0;
    }
    QPushButton.navButton:checked {
        background-color: #7777bb; /* Muted purple/blue for selected */
        color: #ffffff;
        font-weight: bold;
    }
    QLabel#clickableLink {
        color: #007bff; /* Standard link blue */
        text-decoration: underline;
    }
    QLabel#clickableLink:hover {
        color: #0056b3; /* Darker blue on hover */
    }
    QStatusBar {
        background-color: #e0e0e0;
        color: #555;
    }
    QStatusBar::item {
        border: none;
    }
"""

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

def get_inference_model_extensions_filter():
    extensions = [
        "*.pt", "*.pth",      # PyTorch
        "*.onnx",             # ONNX
        "*.torchscript", "*.ptl", # TorchScript
        "*.engine",           # TensorRT
        "*.tflite",           # TensorFlow Lite
        "*.pb",               # TensorFlow Frozen Graph
        "*.mlmodel",          # CoreML
        "*.xml",              # OpenVINO IR (main file)
        "*.param",            # NCNN (main file)
    ]
    return f"Model Files ({' '.join(extensions)});;All Files (*)"


def get_available_cameras():
    index = 0
    arr = []
    # Try MSMF first as it's often more reliable on modern Windows
    preferred_backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] 
    
    for backend in preferred_backends:
        index = 0
        arr = [] # Reset for each backend attempt
        while index < 5: # Check first 5 indices, common for built-in/USB cams
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read() # Try to read a frame
                if ret and frame is not None:
                    arr.append(index)
                cap.release()
            index += 1
        if arr: # If found cameras with this backend, use them
            # print(f"Cameras found using backend {backend}: {arr}")
            return arr
            
    # Fallback if no cameras found with preferred backends
    # print("No cameras found with preferred backends (MSMF, DSHOW). Trying default.")
    index = 0
    arr = []
    while index < 5:
        cap = cv2.VideoCapture(index) # Default backend
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                arr.append(index)
            cap.release()
        index += 1
    # if arr:
        # print(f"Cameras found using default backend: {arr}")
    # else:
        # print("No cameras found with any backend.")
    return arr


def get_icon_path(icon_name, theme):
    """Gets the path to an icon based on the current theme."""
    base_icon_dir = "icons"
    if not os.path.exists(base_icon_dir):
        os.makedirs(base_icon_dir, exist_ok=True)
    if not os.path.exists(os.path.join(base_icon_dir, THEME_LIGHT)):
        os.makedirs(os.path.join(base_icon_dir, THEME_LIGHT), exist_ok=True)
    if not os.path.exists(os.path.join(base_icon_dir, THEME_DARK)):
        os.makedirs(os.path.join(base_icon_dir, THEME_DARK), exist_ok=True)

    path = os.path.join(base_icon_dir, theme, icon_name)
    if os.path.exists(path):
        return path
    fallback_path = os.path.join(base_icon_dir, icon_name)
    if os.path.exists(fallback_path):
        return fallback_path
    return None

# --- Worker Threads ---

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


class InstallSpecificDependencyThread(QThread):
    finished = pyqtSignal(bool, str, str)
    log_message = pyqtSignal(str)

    def __init__(self, dependency_name, model_path_context):
        super().__init__()
        self.dependency_name = dependency_name
        self.model_path_context = model_path_context
        self.pip_package_name = ""

    def run(self):
        dependency_map = {
            "onnxruntime": "onnxruntime",
            "onnxruntime-gpu": "onnxruntime-gpu",
            "openvino": "openvino-dev", # Installs openvino tools as well
            "tflite_runtime": "tflite-runtime",
            "pycoral": "pycoral"
        }
        self.pip_package_name = dependency_map.get(self.dependency_name.lower())

        if not self.pip_package_name:
            self.log_message.emit(f"Don't know how to auto-install: {self.dependency_name}. Please install manually.")
            self.finished.emit(False, f"Unknown dependency for auto-install: {self.dependency_name}", self.model_path_context)
            return

        try:
            self.log_message.emit(f"Attempting to install {self.dependency_name} (pip: {self.pip_package_name})...")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', self.pip_package_name],
                check=True, capture_output=True, text=True
            )
            self.log_message.emit(f"{self.dependency_name} installed successfully via pip.")
            # Attempt a quick import check for some common ones
            if self.dependency_name.lower() in ["onnxruntime", "onnxruntime-gpu"]:
                subprocess.run([sys.executable, "-c", "import onnxruntime"], check=True)
            elif self.dependency_name.lower() == "openvino":
                 subprocess.run([sys.executable, "-c", "from openvino.runtime import Core"], check=True)

            self.finished.emit(True, self.dependency_name, self.model_path_context)
        except subprocess.CalledProcessError as e:
            error_msg = f"Installation of {self.dependency_name} failed: {e.stderr or e.stdout or str(e)}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, error_msg, self.model_path_context)
        except Exception as e:
            error_msg = f"An error occurred during {self.dependency_name} installation: {e}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, error_msg, self.model_path_context)


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
        self.yaml_path = os.path.join(self.final_data_dir, "data.yaml")
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
            
            possible_classes_files = glob.glob(os.path.join(self.custom_data_dir, '**', 'classes.txt'), recursive=True)
            if possible_classes_files:
                classes_file_path = possible_classes_files[0]
                self.log_message.emit(f"Found classes.txt at {classes_file_path}")
            
            for root_dir_to_check in [self.custom_data_dir] + [os.path.join(self.custom_data_dir, d) for d in os.listdir(self.custom_data_dir) if os.path.isdir(os.path.join(self.custom_data_dir, d))]:
                img_d = os.path.join(root_dir_to_check, 'images')
                lbl_d = os.path.join(root_dir_to_check, 'labels')
                if os.path.isdir(img_d): images_src = img_d
                if os.path.isdir(lbl_d): labels_src = lbl_d
                if images_src and labels_src: break
            
            if not images_src:
                self.finished.emit(False, "Could not find 'images' folder. Please ensure it's named 'images' and is in the root or one subfolder of the zip.")
                return
            if not labels_src:
                self.finished.emit(False, "Could not find 'labels' folder. Please ensure it's named 'labels' and is in the root or one subfolder of the zip.")
                return
            if not classes_file_path:
                self.finished.emit(False, "Could not find 'classes.txt'. It must be present in the zip, preferably in the root or alongside images/labels.")
                return

            self.log_message.emit(f"Using images from: {images_src}")
            self.log_message.emit(f"Using labels from: {labels_src}")
            self.progress.emit("Splitting data...", 50)
            all_images = sorted([f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
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
            val_img_path = os.path.join(self.final_data_dir, 'val', 'images') 
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
                'path': os.path.abspath(self.final_data_dir), 
                'train': os.path.join('train', 'images'),      
                'val': os.path.join('val', 'images'),        
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
    finished = pyqtSignal(bool, str) # bool: success, str: path to best.pt or error message

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

            task_type = "segment" if "seg" in self.model_name.lower() else "detect"
            base_run_path = os.path.join(self.project_name, task_type, self.run_name)

            if os.path.exists(base_run_path):
                self.progress_update.emit(f"Deleting existing run directory: {base_run_path}")
                try:
                    shutil.rmtree(base_run_path)
                except OSError as e:
                    self.progress_update.emit(f"Could not delete {base_run_path}: {e}. Training may use an incremented name.")
            
            device_arg = self.device if self.device not in ['cpu', None, ''] else 'cpu'

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device=device_arg,
                project=self.project_name, 
                name=self.run_name, 
                exist_ok=True 
            )
            
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
                 latest_train_dir = str(model.trainer.save_dir)
                 self.progress_update.emit(f"Training results saved in: {latest_train_dir}")
            else: 
                self.progress_update.emit(f"Warning: Could not directly determine exact save directory from model.trainer.save_dir. Searching...")
                pattern = os.path.join(self.project_name, task_type, self.run_name + "*")
                possible_dirs = glob.glob(pattern)
                if not possible_dirs:
                    self.finished.emit(False, "Could not find training results directory.")
                    return
                latest_train_dir = max(possible_dirs, key=os.path.getctime)
                self.progress_update.emit(f"Guessed training results directory: {latest_train_dir}")


            best_pt_path = os.path.join(latest_train_dir, 'weights', 'best.pt')

            if os.path.exists(best_pt_path):
                output_model_dir = "trained_models_gui"
                os.makedirs(output_model_dir, exist_ok=True)
                
                base_model_name_no_ext = os.path.splitext(os.path.basename(self.model_name))[0]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_model_name = f"{base_model_name_no_ext}_custom_{timestamp}.pt"
                final_model_path = os.path.join(output_model_dir, final_model_name)
                
                shutil.copy(best_pt_path, final_model_path)
                self.progress_update.emit(f"Training finished. Best model copied to {final_model_path}")
                self.finished.emit(True, os.path.abspath(final_model_path))
            else:
                self.progress_update.emit(f"Training completed, but 'best.pt' not found in {os.path.join(latest_train_dir, 'weights')}.")
                self.finished.emit(False, f"Training completed, but 'best.pt' not found.")
        except Exception as e:
            self.progress_update.emit(f"Training error: {e}")
            self.finished.emit(False, f"Training failed: {e}")


class InferenceThread(QThread):
    frame_ready = pyqtSignal(QImage, float, int) # QImage, fps, frame_count
    finished = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, model_path, source_type, source_path, device, confidence_threshold=0.25, resolution=None):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type # "webcam", "image", "video"
        self.source_path = source_path # camera index or file path
        self.device = device if device not in [None, ''] else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution # Tuple (width, height) or None
        self._is_running = True
        self.model = None
        self.cap = None
        self.frame_count = 0

    def run(self):
        try:
            self.log_message.emit(f"Attempting to load model for inference: {self.model_path} on device: {self.device}")
            try:
                self.model = YOLO(self.model_path)
                self.log_message.emit(f"Model '{os.path.basename(self.model_path)}' loaded by YOLO constructor.")
            except ImportError as e:
                err_str = str(e).lower()
                dep_name = None
                if "onnxruntime" in err_str: dep_name = "onnxruntime"
                elif "openvino" in err_str: dep_name = "openvino"
                if dep_name:
                    self.log_message.emit(f"ImportError: {dep_name.upper()} is likely missing for {self.model_path}.")
                    self.finished.emit(f"missing_dependency:{dep_name}:{self.model_path}")
                    return
                self.log_message.emit(f"Unhandled ImportError during model load: {e}")
                self.finished.emit(f"Error: Missing library for model - {e}")
                return
            except Exception as e:
                err_str = str(e).lower()
                dep_name = None
                if "onnxruntime" in err_str and ("not found" in err_str or "install" in err_str): dep_name = "onnxruntime"
                elif ("openvino" in err_str or "inference engine" in err_str) and \
                     ("not found" in err_str or "install" in err_str): dep_name = "openvino"
                
                if dep_name:
                    self.log_message.emit(f"Error suggests {dep_name.upper()} is missing for {self.model_path}: {e}")
                    self.finished.emit(f"missing_dependency:{dep_name}:{self.model_path}")
                    return
                self.log_message.emit(f"Failed to load model: {e}")
                self.finished.emit(f"Error: Failed to load model - {e}")
                return

            if self.source_type == "webcam":
                # Try preferred backends for webcam
                camera_opened = False
                preferred_backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
                for backend in preferred_backends:
                    self.cap = cv2.VideoCapture(int(self.source_path), backend)
                    if self.cap.isOpened():
                        camera_opened = True
                        self.log_message.emit(f"Webcam {self.source_path} opened successfully using backend {backend}.")
                        break
                if not camera_opened: # Fallback to default if preferred failed
                    self.cap = cv2.VideoCapture(int(self.source_path))
                    if self.cap.isOpened():
                        camera_opened = True
                        self.log_message.emit(f"Webcam {self.source_path} opened successfully using default backend.")
                
                if not camera_opened:
                    self.log_message.emit(f"Error: Could not open webcam {self.source_path} with any backend.")
                    self.finished.emit("Error: Webcam could not be opened.")
                    return
                
                if self.resolution:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])


            elif self.source_type == "image":
                if not os.path.exists(self.source_path):
                    self.log_message.emit(f"Error: Image file not found at {self.source_path}")
                    self.finished.emit("Error: Image file not found.")
                    return
                frame = cv2.imread(self.source_path)
                if frame is None:
                    self.log_message.emit(f"Error: Could not read image file {self.source_path}")
                    self.finished.emit("Error: Could not read image.")
                    return
                
                self.log_message.emit(f"Processing image: {self.source_path}")
                start_time = time.time()
                results = self.model.predict(source=frame, conf=self.confidence_threshold, device=self.device, verbose=False)
                
                if results and results[0].plot() is not None:
                    processed_frame = results[0].plot() 
                    end_time = time.time()
                    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                    
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_count += 1
                    self.frame_ready.emit(qt_image.copy(), fps, self.frame_count)
                else:
                    self.log_message.emit("No results or plot available for the image.")
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_ready.emit(qt_image.copy(), 0, 1)

                self.finished.emit("Image processing complete.")
                return 

            elif self.source_type == "video":
                if not os.path.exists(self.source_path):
                    self.log_message.emit(f"Error: Video file not found at {self.source_path}")
                    self.finished.emit("Error: Video file not found.")
                    return
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.log_message.emit(f"Error: Could not open video file {self.source_path}.")
                    self.finished.emit("Error: Video file could not be opened.")
                    return
                self.log_message.emit(f"Processing video file: {self.source_path}")

            prev_time = time.time()
            while self._is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.log_message.emit("End of video file or cannot read frame.")
                    break 
                
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time

                results = self.model.predict(source=frame, conf=self.confidence_threshold, device=self.device, verbose=False, stream=False)
                
                processed_frame = frame 
                if results and results[0].plot() is not None:
                    processed_frame = results[0].plot()

                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_count += 1
                self.frame_ready.emit(qt_image.copy(), fps, self.frame_count)

            self.log_message.emit("Inference loop finished.")

        except Exception as e:
            self.log_message.emit(f"Inference error: {e}")
            self.finished.emit(f"Error in inference: {e}")
        finally:
            if self.cap:
                self.cap.release()
                self.log_message.emit("Video capture released.")
            if not self._is_running: 
                 self.finished.emit("Inference stopped by user.")
            elif self.source_type != "image": 
                 self.finished.emit("Inference complete.")


    def stop(self):
        self.log_message.emit("Stopping inference thread...")
        self._is_running = False


class ExportThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str) # success, output_path or error_msg

    def __init__(self, model_path, export_format, device):
        super().__init__()
        self.model_path = model_path
        self.export_format = export_format
        self.device = device if device not in [None, ''] else 'cpu'

    def run(self):
        try:
            self.progress_update.emit(f"Loading model {self.model_path} for export...")
            model = YOLO(self.model_path)
            self.progress_update.emit(f"Exporting to {self.export_format} format on device {self.device}...")
            
            exported_model_path = model.export(format=self.export_format, device=self.device)
            
            self.progress_update.emit(f"Model exported successfully to: {exported_model_path}")
            self.finished.emit(True, exported_model_path)
        except Exception as e:
            self.progress_update.emit(f"Export failed: {e}")
            self.finished.emit(False, str(e))


# --- Main Application Window ---
class ModernYoloGUI(QMainWindow):
    # Store the app version
    APP_VERSION = "1.1.2"  # Updated version number

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionCraft Studio - YOLOv8 GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize core attributes first
        self.current_theme = THEME_LIGHT
        self.log_console = None
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing VisionCraft Studio...")

        # Thread management
        self.inference_thread = None
        self.train_thread = None
        self.data_prep_thread = None
        self.export_thread = None
        self.dependency_install_thread = None
        self.install_ultralytics_thread = None

        # Initialize UI elements to None (will be created in respective methods)
        self.model_path_input = None
        self.confidence_slider = None
        self.resolution_combo = None
        self.device_combo_deploy = None
        self.camera_combo = None
        self.start_webcam_button = None
        self.stop_inference_button = None
        self.video_display_label = None
        self.fps_label = None
        self.toggle_console_button = None
        self.dataset_zip_path_label = None
        self.data_yaml_path_label = None
        self.train_model_combo = None
        self.epochs_spinbox = None
        self.imgsz_spinbox = None
        self.device_combo_train = None
        self.start_train_button = None
        self.train_progress_bar = None
        self.load_trained_to_deploy_checkbox = None
        self.export_model_path_input = None
        self.export_format_combo = None
        self.device_combo_export = None
        self.export_button = None
        self.export_status_label = None
        self.nav_buttons = {}

        # Initialize UI
        self.initUI()
        self.update_theme(THEME_LIGHT)
        self.check_ultralytics_installation()

    def initUI(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.overall_layout = QHBoxLayout(main_widget)

        # Initialize console early
        self.init_console()

        # Create navigation and content areas
        self.create_navigation_bar()
        self.create_content_area()

        # Create pages
        self.create_pages()

        # Add console toggle to status bar
        self.add_console_toggle()

        self.log("Application UI initialized.")

    def init_console(self):
        """Initialize the console log group and text edit."""
        self.log_console_group = QGroupBox("Console Log")
        console_layout = QVBoxLayout(self.log_console_group)
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier", 9))
        console_layout.addWidget(self.log_console)
        self.log_console_group.setFixedHeight(150)
        self.log_console_group.setVisible(True)

    def create_navigation_bar(self):
        """Create the navigation bar on the left side."""
        self.nav_panel = QFrame()
        self.nav_panel.setFixedWidth(200)
        self.nav_panel.setObjectName("navPanel")
        nav_layout = QVBoxLayout(self.nav_panel)
        nav_layout.setAlignment(Qt.AlignTop)
        nav_layout.setSpacing(5)
        self.overall_layout.addWidget(self.nav_panel)

    def create_content_area(self):
        """Create the main content area with stacked widget and console."""
        content_area_widget = QWidget()
        content_and_console_layout = QVBoxLayout(content_area_widget)
        content_and_console_layout.setContentsMargins(0, 0, 0, 0)

        self.content_stack = QStackedWidget()
        content_and_console_layout.addWidget(self.content_stack)
        content_and_console_layout.addWidget(self.log_console_group)

        self.overall_layout.addWidget(content_area_widget)

    def create_pages(self):
        """Create all the pages and their navigation buttons."""
        self.tabs_config = [
            ("Home", "home.png", self.create_home_tab),
            ("Data Prep", "data.png", self.create_data_prep_tab),
            ("Train", "train.png", self.create_train_tab),
            ("Deploy", "deploy.png", self.create_deploy_tab),
            ("Export", "export.png", self.create_export_tab),
            ("Settings", "settings.png", self.create_settings_tab),
            ("About", "about.png", self.create_about_tab),
        ]

        for i, (name, icon_name, creation_func) in enumerate(self.tabs_config):
            page = creation_func()
            self.content_stack.addWidget(page)
            
            button = QPushButton(name)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setFixedHeight(40)
            button.clicked.connect(lambda checked, index=i: self.content_stack.setCurrentIndex(index))
            self.nav_panel.layout().addWidget(button)
            self.nav_buttons[name] = button

        if self.nav_buttons:
            first_button_name = self.tabs_config[0][0]
            self.nav_buttons[first_button_name].setChecked(True)
            self.content_stack.setCurrentIndex(0)

    def add_console_toggle(self):
        """Add console toggle button to status bar."""
        self.toggle_console_button = QPushButton("Toggle Console")
        self.toggle_console_button.setCheckable(True)
        self.toggle_console_button.setChecked(True)
        self.toggle_console_button.setToolTip("Show/Hide Console Log")
        self.toggle_console_button.clicked.connect(self.toggle_console_visibility)
        self.toggle_console_button.setStyleSheet("padding: 3px 7px; font-size: 9pt;")
        self.status_bar.addPermanentWidget(self.toggle_console_button)

    def create_button_with_icon(self, text, icon_name, callback, parent_layout, tooltip=None):
        button = QPushButton(text)
        icon_path = get_icon_path(icon_name, self.current_theme)
        if icon_path:
            button.setIcon(QIcon(icon_path))
            button.setIconSize(ICON_SIZE)
        button.clicked.connect(callback)
        if tooltip:
            button.setToolTip(tooltip)
        parent_layout.addWidget(button)
        return button

    def update_nav_icons(self):
        for name, icon_filename_base, _ in self.tabs_config: # Adjusted to match new tabs_config
            button = self.nav_buttons.get(name)
            if button:
                icon_path = get_icon_path(icon_filename_base, self.current_theme)
                if icon_path:
                    button.setIcon(QIcon(icon_path))
                    button.setIconSize(ICON_SIZE)
                else:
                    button.setIcon(QIcon())
                    # self.log(f"Icon not found for {name} in theme {self.current_theme}") # Avoid logging too much here

    def create_home_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Welcome to VisionCraft Studio!")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        intro_text = QLabel(
            "Your all-in-one solution for preparing data, training YOLOv8 models, "
            "deploying them for real-time inference, and exporting them to various formats.\n\n"
            "Navigate through the tabs on the left to get started."
        )
        intro_text.setWordWrap(True)
        intro_text.setAlignment(Qt.AlignCenter)
        intro_text.setStyleSheet("font-size: 11pt; padding: 20px;")
        layout.addWidget(intro_text)
        
        logo_label = QLabel()
        logo_path = get_icon_path("logo.png", self.current_theme) 
        if logo_path:
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
        else:
            layout.addWidget(QLabel("Imagine a cool logo here! (logo.png missing)"))


        layout.addStretch() 
        return widget

    def create_data_prep_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15) 

        dataset_group = QGroupBox("Dataset Input (Roboflow ZIP format)")
        dataset_layout = QGridLayout(dataset_group) 

        dataset_layout.addWidget(QLabel("Dataset ZIP File:"), 0, 0)
        self.dataset_zip_path_label = QLabel("No file selected.")
        self.dataset_zip_path_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_zip_path_label, 0, 1, 1, 2) 

        browse_zip_button = QPushButton("Browse ZIP")
        browse_zip_button.clicked.connect(self.browse_dataset_zip)
        dataset_layout.addWidget(browse_zip_button, 0, 3)
        
        dataset_layout.addWidget(QLabel("Train/Validation Split (% for Train):"), 1, 0)
        self.train_split_spinbox = QSpinBox()
        self.train_split_spinbox.setRange(10, 90)
        self.train_split_spinbox.setValue(80) 
        self.train_split_spinbox.setSuffix("%")
        dataset_layout.addWidget(self.train_split_spinbox, 1, 1)

        self.start_prep_button = QPushButton("Start Data Preparation")
        self.start_prep_button.clicked.connect(self.start_data_preparation)
        self.start_prep_button.setFixedHeight(35)
        dataset_layout.addWidget(self.start_prep_button, 2, 0, 1, 4) 
        layout.addWidget(dataset_group)

        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel("Generated data.yaml Path:"), 0, 0)
        self.data_yaml_path_label = QLabel("Not generated yet.")
        self.data_yaml_path_label.setWordWrap(True)
        output_layout.addWidget(self.data_yaml_path_label, 0, 1)
        
        self.data_prep_progress_bar = QProgressBar()
        output_layout.addWidget(self.data_prep_progress_bar, 1, 0, 1, 2)
        layout.addWidget(output_group)

        layout.addStretch()
        return widget

    def create_train_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget) 
        main_layout.setSpacing(15)

        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout(config_group) 

        config_layout.addWidget(QLabel("Base Model:"), 0, 0)
        self.train_model_combo = QComboBox()
        self.train_model_combo.addItems(get_yolo_models())
        config_layout.addWidget(self.train_model_combo, 0, 1)

        config_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.train_data_yaml_path_input = QLineEdit()
        self.train_data_yaml_path_input.setPlaceholderText("Path to data.yaml from Data Prep tab or browse")
        config_layout.addWidget(self.train_data_yaml_path_input, 1, 1)
        browse_yaml_button = QPushButton("Browse")
        browse_yaml_button.clicked.connect(self.browse_train_data_yaml)
        config_layout.addWidget(browse_yaml_button, 1, 2)

        config_layout.addWidget(QLabel("Epochs:"), 2, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(50)
        config_layout.addWidget(self.epochs_spinbox, 2, 1)

        config_layout.addWidget(QLabel("Image Size (imgsz):"), 3, 0)
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 2048)
        self.imgsz_spinbox.setSingleStep(32)
        self.imgsz_spinbox.setValue(640)
        config_layout.addWidget(self.imgsz_spinbox, 3, 1)

        config_layout.addWidget(QLabel("Device:"), 4, 0)
        self.device_combo_train = QComboBox()
        self.device_combo_train.addItems(["cpu", "0", "1", "mps"]) 
        self.device_combo_train.setToolTip("Specify GPU device (e.g., 0) or 'cpu' or 'mps' for Apple Silicon")
        self.device_combo_train.setCurrentText(self.detect_best_device())
        config_layout.addWidget(self.device_combo_train, 4, 1)
        
        self.load_trained_to_deploy_checkbox = QCheckBox("Load trained model to Deploy tab after training")
        self.load_trained_to_deploy_checkbox.setChecked(True) 
        config_layout.addWidget(self.load_trained_to_deploy_checkbox, 5, 0, 1, 3) 

        main_layout.addWidget(config_group)

        train_control_group = QGroupBox("Training Control & Progress")
        train_control_layout = QVBoxLayout(train_control_group)

        self.start_train_button = QPushButton("Start Training")
        self.start_train_button.clicked.connect(self.start_training)
        self.start_train_button.setFixedHeight(40) 
        train_control_layout.addWidget(self.start_train_button)

        self.train_progress_bar = QProgressBar()
        train_control_layout.addWidget(self.train_progress_bar)
        
        self.trained_model_path_label = QLabel("Trained model will appear here.")
        self.trained_model_path_label.setWordWrap(True)
        train_control_layout.addWidget(self.trained_model_path_label)

        main_layout.addWidget(train_control_group)
        main_layout.addStretch()
        return widget

    def create_deploy_tab(self):
        widget = QWidget()
        main_h_layout = QHBoxLayout(widget)

        controls_frame = QFrame()
        controls_frame.setFixedWidth(350) 
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(10)
        controls_layout.setAlignment(Qt.AlignTop)

        model_group = QGroupBox("Model & Source")
        model_layout = QGridLayout(model_group)

        model_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to .pt, .onnx, etc.")
        model_layout.addWidget(self.model_path_input, 0, 1, 1, 2) 
        browse_model_button = QPushButton("Browse")
        browse_model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_model_button, 0, 3)

        model_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence_slider = QSpinBox() 
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setSuffix("%")
        model_layout.addWidget(self.confidence_slider, 1, 1, 1, 3) 

        model_layout.addWidget(QLabel("Resolution (Webcam):"), 2, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Default", "640x480", "1280x720", "1920x1080"])
        model_layout.addWidget(self.resolution_combo, 2, 1, 1, 3)

        model_layout.addWidget(QLabel("Device:"), 3, 0)
        self.device_combo_deploy = QComboBox()
        self.device_combo_deploy.addItems(["cpu", "0", "1", "mps"])
        self.device_combo_deploy.setCurrentText(self.detect_best_device())
        model_layout.addWidget(self.device_combo_deploy, 3, 1, 1, 3)
        
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_inference_model)
        model_layout.addWidget(self.load_model_button, 4, 0, 1, 4) 
        controls_layout.addWidget(model_group)


        source_group = QGroupBox("Inference Source")
        source_layout = QGridLayout(source_group)

        source_layout.addWidget(QLabel("Webcam:"), 0, 0)
        self.camera_combo = QComboBox()
        self.populate_camera_combo() # This call is now safe as self.log_console exists
        source_layout.addWidget(self.camera_combo, 0, 1)

        self.start_webcam_button = QPushButton("Start Webcam") 
        self.start_webcam_button.setCheckable(False) 
        self.start_webcam_button.clicked.connect(self.toggle_webcam_inference)
        source_layout.addWidget(self.start_webcam_button, 0, 2)

        upload_image_button = QPushButton("Upload Image")
        upload_image_button.clicked.connect(lambda: self.start_inference_from_file("image"))
        source_layout.addWidget(upload_image_button, 1, 0, 1, 3) 

        upload_video_button = QPushButton("Upload Video")
        upload_video_button.clicked.connect(lambda: self.start_inference_from_file("video"))
        source_layout.addWidget(upload_video_button, 2, 0, 1, 3) 
        
        self.stop_inference_button = QPushButton("Stop Inference")
        self.stop_inference_button.clicked.connect(self.stop_current_inference)
        self.stop_inference_button.setEnabled(False) 
        self.stop_inference_button.setFixedHeight(30)
        source_layout.addWidget(self.stop_inference_button, 3, 0, 1, 3)

        controls_layout.addWidget(source_group)
        controls_layout.addStretch()
        main_h_layout.addWidget(controls_frame)

        video_display_frame = QFrame()
        video_display_layout = QVBoxLayout(video_display_frame)

        self.video_display_label = QLabel("Live feed / Processed image will appear here.")
        self.video_display_label.setObjectName("video_display_label") 
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setMinimumSize(640, 480) 
        self.video_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_display_layout.addWidget(self.video_display_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignRight)
        video_display_layout.addWidget(self.fps_label)

        main_h_layout.addWidget(video_display_frame)
        return widget

    def create_export_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setAlignment(Qt.AlignTop) 

        export_config_group = QGroupBox("Export Configuration")
        export_config_layout = QGridLayout(export_config_group)

        export_config_layout.addWidget(QLabel("Model to Export (.pt):"), 0, 0)
        self.export_model_path_input = QLineEdit()
        self.export_model_path_input.setPlaceholderText("Path to trained .pt model")
        export_config_layout.addWidget(self.export_model_path_input, 0, 1)
        browse_export_model_button = QPushButton("Browse")
        browse_export_model_button.clicked.connect(self.browse_export_model)
        export_config_layout.addWidget(browse_export_model_button, 0, 2)

        export_config_layout.addWidget(QLabel("Export Format:"), 1, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(get_export_formats())
        export_config_layout.addWidget(self.export_format_combo, 1, 1, 1, 2) 

        export_config_layout.addWidget(QLabel("Device for Export:"), 2, 0)
        self.device_combo_export = QComboBox()
        self.device_combo_export.addItems(["cpu", "0", "1", "mps"]) 
        self.device_combo_export.setCurrentText(self.detect_best_device())
        export_config_layout.addWidget(self.device_combo_export, 2, 1, 1, 2)

        layout.addWidget(export_config_group)

        self.export_button = QPushButton("Start Export")
        self.export_button.clicked.connect(self.start_export)
        self.export_button.setFixedHeight(35)
        layout.addWidget(self.export_button)

        self.export_status_label = QLabel("Export status will appear here.")
        self.export_status_label.setWordWrap(True)
        layout.addWidget(self.export_status_label)
        
        self.export_progress_bar = QProgressBar() 
        layout.addWidget(self.export_progress_bar)


        layout.addStretch()
        return widget

    def create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop) 
        layout.setSpacing(15)

        theme_group = QGroupBox("Appearance")
        theme_layout = QHBoxLayout(theme_group)
        theme_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        current_theme_text = "Dark" if self.current_theme == THEME_DARK else "Light"
        self.theme_combo.setCurrentText(current_theme_text)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        layout.addWidget(theme_group)
        
        ultralytics_group = QGroupBox("Ultralytics Installation")
        ultralytics_layout = QVBoxLayout(ultralytics_group)
        self.ultralytics_status_label = QLabel("Checking Ultralytics installation...")
        ultralytics_layout.addWidget(self.ultralytics_status_label)
        self.install_ultralytics_button = QPushButton("Install/Reinstall Ultralytics")
        self.install_ultralytics_button.clicked.connect(self.prompt_install_ultralytics)
        ultralytics_layout.addWidget(self.install_ultralytics_button)
        layout.addWidget(ultralytics_group)


        layout.addStretch() 
        return widget

    def create_about_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        title_label = QLabel("VisionCraft Studio")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        version_label = QLabel("Version 1.1.1") # Incremented version for this fix
        version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(version_label)

        description = QLabel(
            "A comprehensive tool for working with YOLOv8 models, "
            "from data preparation to deployment and export.\n"
            "Built with Python, PyQt5, and Ultralytics YOLO."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("padding: 0 20px;") 
        layout.addWidget(description)

        dev_intro_label = QLabel("Developed by:")
        dev_intro_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(dev_intro_label)

        self.developer_name_label = QLabel("CodeWithSalman") 
        self.developer_name_label.setObjectName("clickableLink")
        self.developer_name_label.setAlignment(Qt.AlignCenter)
        self.developer_name_label.setToolTip("Visit developer's website (Opens in browser)")
        # Replace with your actual URL
        self.developer_name_label.mousePressEvent = lambda event, url="YOUR_YOUTUBE_CHANNEL_URL_HERE": self.open_link(url) 
        layout.addWidget(self.developer_name_label)

        self.buy_coffee_label = QLabel("Support the Developer - Buy Me a Coffee!")
        self.buy_coffee_label.setObjectName("clickableLink")
        self.buy_coffee_label.setAlignment(Qt.AlignCenter)
        self.buy_coffee_label.setToolTip("Support via Buy Me a Coffee (Opens in browser)")
         # Replace with your actual URL
        self.buy_coffee_label.mousePressEvent = lambda event, url="YOUR_BUYMECOFFEE_URL_HERE": self.open_link(url)
        layout.addWidget(self.buy_coffee_label)


        layout.addStretch()
        return widget

    def open_link(self, url_string):
        if url_string == "YOUR_YOUTUBE_CHANNEL_URL_HERE" or url_string == "YOUR_BUYMECOFFEE_URL_HERE":
            self.log(f"Placeholder link clicked. Please update the URL in create_about_tab for: {url_string}")
            QMessageBox.information(self, "Placeholder Link", "This link is a placeholder. The developer needs to update it in the code.")
            return

        url = QUrl(url_string)
        if not QDesktopServices.openUrl(url):
            self.log(f"Error: Could not open URL {url_string}")
            QMessageBox.warning(self, "Open Link Error", f"Could not open the link: {url_string}")


    # --- Theming and Icon Logic ---
    def on_theme_changed(self, theme_text):
        new_theme = THEME_DARK if theme_text == "Dark" else THEME_LIGHT
        self.update_theme(new_theme)

    def update_theme(self, theme):
        self.current_theme = theme
        if theme == THEME_DARK:
            self.setStyleSheet(DARK_STYLESHEET)
        else:
            self.setStyleSheet(LIGHT_STYLESHEET) 
        self.update_nav_icons() 
        self.log(f"Theme changed to {theme}.")
        if self.toggle_console_button:
            icon_path = get_icon_path("console_toggle.png", self.current_theme) 
            if icon_path:
                self.toggle_console_button.setIcon(QIcon(icon_path))
            else:
                self.toggle_console_button.setText("↕️ Console") 


    # --- Console and Logging ---
    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Check if log_console and status_bar are initialized before using them
        if self.log_console is not None:
            self.log_console.append(f"[{timestamp}] {message}")
        else:
            # Fallback to print if log_console is not yet available (shouldn't happen with new init order)
            print(f"LOG_FALLBACK: [{timestamp}] {message}")

        if self.status_bar is not None:
            self.status_bar.showMessage(message, 5000)
        else:
            # Fallback for status bar
            print(f"STATUS_FALLBACK: {message}")


    def toggle_console_visibility(self):
        if self.log_console_group: # Check if it exists
            is_visible = self.log_console_group.isVisible()
            self.log_console_group.setVisible(not is_visible)
            if self.toggle_console_button: # Check if button exists
                 self.toggle_console_button.setChecked(not is_visible)


    # --- Ultralytics Checks and Installation ---
    def check_ultralytics_installation(self):
        if ULTRALYTICS_INSTALLED:
            try:
                _ = YOLO('yolov8n.pt') 
                self.log("Ultralytics YOLO is installed and accessible.")
                if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label:
                     self.ultralytics_status_label.setText("Ultralytics is installed and working.")
                if hasattr(self, 'install_ultralytics_button') and self.install_ultralytics_button:
                     self.install_ultralytics_button.setText("Reinstall Ultralytics (Optional)")
                return True
            except Exception as e:
                self.log(f"Ultralytics seems installed but failed a quick check: {e}")
                if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label:
                    self.ultralytics_status_label.setText(f"Ultralytics installed but check failed: {e}")
                if hasattr(self, 'install_ultralytics_button') and self.install_ultralytics_button:
                    self.install_ultralytics_button.setText("Attempt Reinstall Ultralytics")
                QMessageBox.warning(self, "Ultralytics Check Failed", 
                                    f"Ultralytics is imported, but a quick test failed: {e}\n"
                                    "Some operations might not work correctly. Consider reinstalling from the Settings tab.")
                return False 
        else:
            self.log("Ultralytics YOLO is not installed.")
            if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label:
                self.ultralytics_status_label.setText("Ultralytics is NOT installed. Please install it.")
            if hasattr(self, 'install_ultralytics_button') and self.install_ultralytics_button:
                self.install_ultralytics_button.setText("Install Ultralytics")
            self.prompt_install_ultralytics()
            return False

    def prompt_install_ultralytics(self):
        if self.install_ultralytics_thread and self.install_ultralytics_thread.isRunning():
            self.log("Ultralytics installation is already in progress.")
            return

        reply = QMessageBox.question(self, "Install Ultralytics",
                                     "Ultralytics YOLO library is required. Do you want to install it now? "
                                     "This may take a few minutes.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.log("Starting Ultralytics installation...")
            if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label: self.ultralytics_status_label.setText("Installing Ultralytics...")
            self.install_ultralytics_thread = InstallThread()
            self.install_ultralytics_thread.log_message.connect(self.log)
            self.install_ultralytics_thread.finished.connect(self.on_ultralytics_install_finished)
            self.install_ultralytics_thread.start()
        else:
            self.log("Ultralytics installation declined by user.")
            if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label: self.ultralytics_status_label.setText("Ultralytics installation declined.")


    def on_ultralytics_install_finished(self, success, message):
        self.log(f"Ultralytics installation finished: {message}")
        if success:
            QMessageBox.information(self, "Installation Success", 
                                    "Ultralytics installed successfully! Please restart the application for changes to take full effect if this was the first install.")
            if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label: self.ultralytics_status_label.setText("Ultralytics installed successfully. Restart may be needed.")
            if hasattr(self, 'install_ultralytics_button') and self.install_ultralytics_button: self.install_ultralytics_button.setText("Reinstall Ultralytics (Optional)")
            global ULTRALYTICS_INSTALLED
            ULTRALYTICS_INSTALLED = True 
        else:
            QMessageBox.critical(self, "Installation Failed", f"Ultralytics installation failed: {message}")
            if hasattr(self, 'ultralytics_status_label') and self.ultralytics_status_label: self.ultralytics_status_label.setText(f"Ultralytics installation failed: {message}")
            if hasattr(self, 'install_ultralytics_button') and self.install_ultralytics_button: self.install_ultralytics_button.setText("Retry Install Ultralytics")


    # --- Data Prep Tab Logic ---
    def browse_dataset_zip(self):
        zip_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset ZIP", "", "ZIP Files (*.zip)")
        if zip_path:
            if self.dataset_zip_path_label: self.dataset_zip_path_label.setText(zip_path)
            self.log(f"Dataset ZIP selected: {zip_path}")

    def start_data_preparation(self):
        if self.data_prep_thread and self.data_prep_thread.isRunning():
            self.log("Data preparation is already in progress.")
            return

        zip_path = self.dataset_zip_path_label.text() if self.dataset_zip_path_label else None
        if not zip_path or zip_path == "No file selected.":
            QMessageBox.warning(self, "Missing Input", "Please select a dataset ZIP file.")
            return

        if not ULTRALYTICS_INSTALLED and not self.check_ultralytics_installation():
             self.log("Ultralytics not available. Data prep cannot start.")
             return

        self.log("Starting data preparation...")
        if self.data_prep_progress_bar: self.data_prep_progress_bar.setValue(0)
        if self.data_yaml_path_label: self.data_yaml_path_label.setText("Processing...")
        
        app_root = os.getcwd() 
        data_dir_base = os.path.join(app_root, "datasets_prepared")
        os.makedirs(data_dir_base, exist_ok=True)

        train_pct = self.train_split_spinbox.value() / 100.0 if self.train_split_spinbox else 0.8

        self.data_prep_thread = DataPrepThread(zip_path, data_dir_base, train_pct)
        self.data_prep_thread.log_message.connect(self.log)
        self.data_prep_thread.progress.connect(self.on_data_prep_progress)
        self.data_prep_thread.finished.connect(self.on_data_prep_finished)
        if self.start_prep_button: self.start_prep_button.setEnabled(False)
        self.data_prep_thread.start()

    def on_data_prep_progress(self, message, value):
        self.log(f"Data Prep: {message}")
        if self.data_prep_progress_bar: self.data_prep_progress_bar.setValue(value)

    def on_data_prep_finished(self, success, result_path_or_message):
        if self.start_prep_button: self.start_prep_button.setEnabled(True)
        if success:
            self.log(f"Data preparation successful. YAML created at: {result_path_or_message}")
            if self.data_yaml_path_label: self.data_yaml_path_label.setText(result_path_or_message)
            if self.train_data_yaml_path_input: self.train_data_yaml_path_input.setText(result_path_or_message)
            QMessageBox.information(self, "Data Prep Complete", f"Dataset prepared successfully!\ndata.yaml: {result_path_or_message}")
        else:
            self.log(f"Data preparation failed: {result_path_or_message}")
            if self.data_yaml_path_label: self.data_yaml_path_label.setText("Failed.")
            if self.data_prep_progress_bar: self.data_prep_progress_bar.setValue(0)
            QMessageBox.critical(self, "Data Prep Error", f"Data preparation failed: {result_path_or_message}")

    # --- Train Tab Logic ---
    def browse_train_data_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml)")
        if yaml_path:
            if self.train_data_yaml_path_input: self.train_data_yaml_path_input.setText(yaml_path)
            self.log(f"Training data.yaml selected: {yaml_path}")

    def start_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.log("Training is already in progress.")
            return

        if not ULTRALYTICS_INSTALLED and not self.check_ultralytics_installation():
             self.log("Ultralytics not available. Training cannot start.")
             return

        model_name = self.train_model_combo.currentText() if self.train_model_combo else get_yolo_models()[0]
        data_yaml = self.train_data_yaml_path_input.text() if self.train_data_yaml_path_input else ""
        epochs = self.epochs_spinbox.value() if self.epochs_spinbox else 50
        imgsz = self.imgsz_spinbox.value() if self.imgsz_spinbox else 640
        device = self.device_combo_train.currentText() if self.device_combo_train else self.detect_best_device()

        if not data_yaml or not os.path.exists(data_yaml):
            QMessageBox.warning(self, "Missing Input", "Please provide a valid path to data.yaml.")
            return

        self.log(f"Starting training: Model={model_name}, Data={data_yaml}, Epochs={epochs}, ImgSz={imgsz}, Device={device}")
        if self.train_progress_bar: self.train_progress_bar.setValue(0) 
        if self.trained_model_path_label: self.trained_model_path_label.setText("Training in progress...")
        if self.start_train_button: self.start_train_button.setEnabled(False) 

        self.train_thread = TrainThread(model_name, data_yaml, epochs, imgsz, device)
        self.train_thread.progress_update.connect(self.on_train_progress)
        self.train_thread.finished.connect(self.on_train_finished)
        self.train_thread.start()

    def on_train_progress(self, message):
        self.log(f"Train: {message}")
        if self.train_progress_bar:
            current_val = self.train_progress_bar.value()
            if current_val < 95: 
                self.train_progress_bar.setValue(current_val + 1)


    def on_train_finished(self, success, result_path_or_message):
        if self.start_train_button: self.start_train_button.setEnabled(True) 
        if self.train_progress_bar: self.train_progress_bar.setValue(100 if success else 0)
        if success:
            self.log(f"Training successful. Best model saved at: {result_path_or_message}")
            if self.trained_model_path_label: self.trained_model_path_label.setText(f"Best model: {result_path_or_message}")
            QMessageBox.information(self, "Training Complete", f"Training finished successfully!\nBest model: {result_path_or_message}")
            
            if self.load_trained_to_deploy_checkbox and self.load_trained_to_deploy_checkbox.isChecked():
                if self.model_path_input: self.model_path_input.setText(result_path_or_message)
                self.log(f"Trained model path '{result_path_or_message}' loaded into Deploy tab.")
                
            if self.export_model_path_input:
                self.export_model_path_input.setText(result_path_or_message)
                self.log(f"Trained model path '{result_path_or_message}' loaded into Export tab.")

        else:
            self.log(f"Training failed: {result_path_or_message}")
            if self.trained_model_path_label: self.trained_model_path_label.setText(f"Training failed: {result_path_or_message}")
            QMessageBox.critical(self, "Training Error", f"Training failed: {result_path_or_message}")

    # --- Deploy Tab Logic ---
    def populate_camera_combo(self):
        if not self.camera_combo: return # Guard if called too early (though init order should prevent)
        self.camera_combo.clear()
        available_cameras = get_available_cameras()
        if available_cameras:
            self.camera_combo.addItems([str(cam_idx) for cam_idx in available_cameras])
            self.log(f"Available cameras: {available_cameras}") # This log is now safe
        else:
            self.log("No cameras found.") # This log is now safe
            self.camera_combo.addItem("No cameras found")
            self.camera_combo.setEnabled(False)


    def browse_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", get_inference_model_extensions_filter())
        if model_path:
            if self.model_path_input: self.model_path_input.setText(model_path)
            self.log(f"Inference model selected: {model_path}")

    def load_inference_model(self):
        model_path = self.model_path_input.text() if self.model_path_input else ""
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Missing Model", "Please select a valid model file first.")
            return False

        if not ULTRALYTICS_INSTALLED and not self.check_ultralytics_installation():
             self.log("Ultralytics not available. Cannot load model.")
             return False
        
        self.log(f"Attempting to preload model: {model_path} for inference configuration.")
        try:
            _ = YOLO(model_path) 
            self.log(f"Model {model_path} seems valid for YOLO. Ready for inference.")
            QMessageBox.information(self, "Model Ready", f"Model '{os.path.basename(model_path)}' is ready. You can now start webcam or upload image/video.")
            if self.start_webcam_button: self.start_webcam_button.setEnabled(True)
            return True 
        except ImportError as e: 
            err_str = str(e).lower()
            dep_name = None
            if "onnxruntime" in err_str: dep_name = "onnxruntime"
            elif "openvino" in err_str: dep_name = "openvino"
            if dep_name:
                self.handle_missing_dependency(dep_name, model_path)
            else:
                self.log(f"Failed to load model due to missing library: {e}")
                QMessageBox.critical(self, "Model Load Error", f"Failed to load model: Missing library {e}")
            return False
        except Exception as e:
            self.log(f"Failed to load model {model_path}: {e}")
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model: {e}")
            err_str = str(e).lower()
            dep_name = None
            if "onnxruntime" in err_str and ("not found" in err_str or "install" in err_str): dep_name = "onnxruntime"
            elif ("openvino" in err_str or "inference engine" in err_str) and \
                 ("not found" in err_str or "install" in err_str): dep_name = "openvino"
            if dep_name:
                self.handle_missing_dependency(dep_name, model_path)
            return False

    def toggle_webcam_inference(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Stopping webcam inference...")
            self.stop_current_inference() 
        else:
            self.log("Starting webcam inference...")
            if not self.load_inference_model(): 
                self.log("Model not loaded or invalid. Webcam cannot start.")
                return

            if not self.camera_combo or self.camera_combo.currentText() == "No cameras found" or not self.camera_combo.currentText():
                QMessageBox.warning(self, "No Camera", "No camera selected or available.")
                return

            source_path = self.camera_combo.currentText()
            self._start_common_inference("webcam", source_path)
            if self.inference_thread and self.inference_thread.isRunning():
                 if self.start_webcam_button: self.start_webcam_button.setText("Stop Webcam")


    def start_inference_from_file(self, source_type): 
        if not self.load_inference_model(): 
            self.log(f"Model not loaded or invalid. Cannot process {source_type}.")
            return

        if source_type == "image":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)")
        elif source_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        else:
            return

        if file_path:
            self.log(f"Starting inference for {source_type}: {file_path}")
            self._start_common_inference(source_type, file_path)


    def _start_common_inference(self, source_type, source_path):
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(self, "Inference Running",
                                         "An inference process is already running. Stop it and start the new one?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.stop_current_inference()
                QTimer.singleShot(100, lambda: self._proceed_with_inference(source_type, source_path))
            else:
                return 
        else:
            self._proceed_with_inference(source_type, source_path)


    def _proceed_with_inference(self, source_type, source_path):
        model_path = self.model_path_input.text() if self.model_path_input else ""
        device = self.device_combo_deploy.currentText() if self.device_combo_deploy else self.detect_best_device()
        confidence = (self.confidence_slider.value() / 100.0) if self.confidence_slider else 0.25
        
        selected_res_text = self.resolution_combo.currentText() if self.resolution_combo else "Default"
        resolution = None
        if selected_res_text != "Default":
            try:
                w, h = map(int, selected_res_text.split('x'))
                resolution = (w, h)
            except ValueError:
                self.log(f"Invalid resolution format: {selected_res_text}. Using default.")

        self.log(f"Initiating inference: Type={source_type}, Path={source_path}, Model={model_path}, Conf={confidence}, Device={device}, Res={resolution}")
        if self.video_display_label: self.video_display_label.setText(f"Loading {source_type}...") 
        if self.fps_label: self.fps_label.setText("FPS: --")

        self.inference_thread = InferenceThread(model_path, source_type, source_path, device, confidence, resolution)
        self.inference_thread.log_message.connect(self.log)
        self.inference_thread.frame_ready.connect(self.display_frame)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.start()

        if self.stop_inference_button: self.stop_inference_button.setEnabled(True)
        if source_type == "webcam":
            if self.start_webcam_button:
                self.start_webcam_button.setText("Stop Webcam")
                self.start_webcam_button.setEnabled(True) 
        else: 
            if self.start_webcam_button: self.start_webcam_button.setEnabled(False) 


    def stop_current_inference(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Stop signal sent to inference thread.")
            self.inference_thread.stop()
        else:
            self.log("No active inference thread to stop.")
            self._reset_inference_ui_state() 

    def _reset_inference_ui_state(self):
        if self.stop_inference_button: self.stop_inference_button.setEnabled(False)
        if self.start_webcam_button:
            self.start_webcam_button.setText("Start Webcam")
            self.start_webcam_button.setEnabled(True) 
        if self.fps_label: self.fps_label.setText("FPS: --")


    def display_frame(self, q_image, fps, frame_count):
        if self.video_display_label:
            scaled_image = q_image.scaled(self.video_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_display_label.setPixmap(QPixmap.fromImage(scaled_image))
        if self.fps_label: self.fps_label.setText(f"FPS: {fps:.2f} | Frames: {frame_count}")

    def on_inference_finished(self, message):
        self.log(f"Inference finished/stopped: {message}")
        self._reset_inference_ui_state() 

        if message.startswith("missing_dependency:"):
            parts = message.split(':', 2) # Split max 2 times
            if len(parts) == 3: 
                dep_name = parts[1]
                model_path_context = parts[2]
                self.handle_missing_dependency(dep_name, model_path_context)
            else:
                QMessageBox.warning(self, "Inference Error", f"Inference failed due to an unspecified missing dependency from message: {message}")
        elif "Error" in message or "failed" in message: 
            QMessageBox.warning(self, "Inference Error", f"Inference process encountered an error: {message}")
        elif message == "Image processing complete.": 
             self.log("Single image processing is complete.")
        elif message == "Inference stopped by user.":
            self.log("Inference was stopped by the user.")
        else: 
            self.log(f"Inference process ended: {message}")


    def handle_missing_dependency(self, dependency_name, model_path_context):
        self.log(f"Missing dependency: {dependency_name} for model {model_path_context}")
        pip_package = dependency_name.lower()
        if dependency_name.lower() == "openvino": pip_package = "openvino-dev"
        
        reply = QMessageBox.question(self, f"Missing Dependency: {dependency_name.upper()}",
                                     f"The model '{os.path.basename(model_path_context)}' requires '{dependency_name}'.\n"
                                     f"Do you want to attempt to install it now?\n"
                                     f"(This will try: pip install {pip_package})",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            if self.dependency_install_thread and self.dependency_install_thread.isRunning():
                self.log("Another dependency installation is already in progress.")
                return

            self.log(f"Starting installation of {dependency_name}...")
            self.dependency_install_thread = InstallSpecificDependencyThread(dependency_name, model_path_context)
            self.dependency_install_thread.log_message.connect(self.log)
            self.dependency_install_thread.finished.connect(self.on_specific_dependency_install_finished)
            self.dependency_install_thread.start()
        else:
            self.log(f"User declined to install {dependency_name}.")


    def on_specific_dependency_install_finished(self, success, dep_name_or_error, model_path_context_or_dep_name):
        # The arguments might be (success, dep_name, model_path_context) on success
        # or (success, error_message, model_path_context) on failure from thread.
        if success:
            dep_name = dep_name_or_error
            original_model_path = model_path_context_or_dep_name
            self.log(f"{dep_name} installed successfully.")
            QMessageBox.information(self, "Installation Success",
                                    f"{dep_name} installed successfully. Please try loading the model '{os.path.basename(original_model_path)}' again.")
        else:
            error_message = dep_name_or_error
            # If the thread passed dep_name as the third arg on failure, use it.
            dep_name = model_path_context_or_dep_name if isinstance(model_path_context_or_dep_name, str) and not os.path.exists(model_path_context_or_dep_name) else "Dependency"
            self.log(f"Failed to install {dep_name}: {error_message}")
            QMessageBox.critical(self, "Installation Failed",
                                 f"Failed to install {dep_name}. Please install it manually and restart the application.\nError: {error_message}")


    # --- Export Tab Logic ---
    def browse_export_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model to Export", "", "PyTorch Model Files (*.pt)")
        if model_path:
            if self.export_model_path_input: self.export_model_path_input.setText(model_path)
            self.log(f"Model for export selected: {model_path}")

    def start_export(self):
        if self.export_thread and self.export_thread.isRunning():
            self.log("Export is already in progress.")
            return
        
        if not ULTRALYTICS_INSTALLED and not self.check_ultralytics_installation():
             self.log("Ultralytics not available. Export cannot start.")
             return

        model_path = self.export_model_path_input.text() if self.export_model_path_input else ""
        export_format = self.export_format_combo.currentText() if self.export_format_combo else get_export_formats()[0]
        device = self.device_combo_export.currentText() if self.device_combo_export else self.detect_best_device()

        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Missing Input", "Please select a valid .pt model file to export.")
            return

        self.log(f"Starting export: Model={model_path}, Format={export_format}, Device={device}")
        if self.export_status_label: self.export_status_label.setText(f"Exporting to {export_format}...")
        if self.export_button: self.export_button.setEnabled(False)
        if self.export_progress_bar: self.export_progress_bar.setValue(0) 

        self.export_thread = ExportThread(model_path, export_format, device)
        self.export_thread.progress_update.connect(self.on_export_progress)
        self.export_thread.finished.connect(self.on_export_finished)
        self.export_thread.start()

    def on_export_progress(self, message):
        self.log(f"Export: {message}")
        if self.export_status_label: self.export_status_label.setText(message)
        if self.export_progress_bar:
            current_val = self.export_progress_bar.value()
            if current_val < 95 : self.export_progress_bar.setValue(current_val + 5)


    def on_export_finished(self, success, result_path_or_message):
        if self.export_button: self.export_button.setEnabled(True)
        if self.export_progress_bar: self.export_progress_bar.setValue(100 if success else 0)
        if success:
            self.log(f"Export successful. Saved to: {result_path_or_message}")
            if self.export_status_label: self.export_status_label.setText(f"Exported successfully: {result_path_or_message}")
            QMessageBox.information(self, "Export Complete", f"Model exported successfully!\nPath: {result_path_or_message}")
            if self.model_path_input and result_path_or_message.endswith((".onnx", ".engine", ".tflite", ".pb")): 
                self.model_path_input.setText(result_path_or_message)
                self.log(f"Exported model '{result_path_or_message}' also set in Deploy tab.")
        else:
            self.log(f"Export failed: {result_path_or_message}")
            if self.export_status_label: self.export_status_label.setText(f"Export failed: {result_path_or_message}")
            QMessageBox.critical(self, "Export Error", f"Export failed: {result_path_or_message}")


    def detect_best_device(self):
        if torch.cuda.is_available():
            return "0" 
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): 
            return "mps"
        return "cpu"

    def closeEvent(self, event):
        self.log("Application closing. Stopping active threads...")
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait(2000) 

        if self.train_thread and self.train_thread.isRunning():
            self.log("Attempting to quit training thread (might be abrupt)...")
            self.train_thread.quit() 
            self.train_thread.wait(1000)

        if self.data_prep_thread and self.data_prep_thread.isRunning():
            self.data_prep_thread.quit()
            self.data_prep_thread.wait(1000)

        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait(1000)
        
        if self.dependency_install_thread and self.dependency_install_thread.isRunning():
             self.dependency_install_thread.quit()
             self.dependency_install_thread.wait(1000)
        if self.install_ultralytics_thread and self.install_ultralytics_thread.isRunning():
            self.install_ultralytics_thread.quit()
            self.install_ultralytics_thread.wait(1000)

        self.log("Exiting VisionCraft Studio.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    dummy_icon_dir_base = "icons"
    dummy_themes = [THEME_LIGHT, THEME_DARK]
    dummy_icon_names = ["home.png", "data.png", "train.png", "deploy.png", "export.png", "settings.png", "about.png", "logo.png", "console_toggle.png", "arrow_down.png"]

    for theme_name in dummy_themes:
        theme_dir = os.path.join(dummy_icon_dir_base, theme_name)
        if not os.path.exists(theme_dir):
            os.makedirs(theme_dir, exist_ok=True)
        
        for icon_name in dummy_icon_names:
            icon_path = os.path.join(theme_dir, icon_name)
            if not os.path.exists(icon_path):
                try:
                    img = QImage(32, 32, QImage.Format_ARGB32) 
                    img.fill(Qt.transparent) 
                    from PyQt5.QtGui import QPainter, QColor, QBrush # Local import for this block
                    painter = QPainter(img)
                    if theme_name == THEME_DARK:
                        painter.setBrush(QBrush(QColor(DRACULA_PURPLE)))
                    else:
                        painter.setBrush(QBrush(QColor(Qt.gray))) 
                    painter.drawEllipse(4, 4, 24, 24) 
                    painter.end()
                    img.save(icon_path)
                except Exception as e:
                    print(f"Could not create dummy icon {icon_path}: {e}")

    window = ModernYoloGUI()
    window.show()
    sys.exit(app.exec_())
