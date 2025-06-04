import sys
import os
import subprocess
import zipfile
import shutil
import yaml
import cv2
import torch
import numpy as np  # For dummy frame in dependency check
import random
import glob
import time
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLabel,
    QSpinBox,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QFrame,
    QGridLayout,
    QCheckBox,
    QMessageBox,
    QSpacerItem,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont

# Try to import ultralytics, prompt for installation if not found
try:
    from ultralytics import YOLO
    from ultralytics.utils.checks import (
        check_requirements,
    )  # For specific checks if needed

    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False


# --- Utility Functions ---
def get_yolo_models():
    return [
        # YOLOv8 Models
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt",
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
        # YOLOv9 Models
        "yolov9c.pt",
        "yolov9e.pt",
        "yolov9c-seg.pt",
        "yolov9e-seg.pt",
        # # YOLOv10 Models
        # "yolov10n.pt",
        # "yolov10s.pt",
        # "yolov10m.pt",
        # "yolov10l.pt",
        # "yolov10x.pt",
        # "yolov10n-seg.pt",
        # "yolov10s-seg.pt",
        # "yolov10m-seg.pt",
        # "yolov10l-seg.pt",
        # "yolov10x-seg.pt",
        # # YOLOv11 Models
        # "yolov11n.pt",
        # "yolov11s.pt",
        # "yolov11m.pt",
        # "yolov11l.pt",
        # "yolov11x.pt",
        # "yolov11n-seg.pt",
        # "yolov11s-seg.pt",
        # "yolov11m-seg.pt",
        # "yolov11l-seg.pt",
        # "yolov11x-seg.pt",
        # # YOLOv12 Models
        # "yolov12n.pt",
        # "yolov12s.pt",
        # "yolov12m.pt",
        # "yolov12l.pt",
        # "yolov12x.pt",
        # "yolov12n-seg.pt",
        # "yolov12s-seg.pt",
        # "yolov12m-seg.pt",
        # "yolov12l-seg.pt",
        # "yolov12x-seg.pt",
    ]


def get_export_formats():
    return [
        "onnx",
        "torchscript",
        "coreml",
        "saved_model",
        "pb",
        "tflite",
        "edgetpu",
        "tfjs",
        "paddle",
        "ncnn",
        "openvino",
    ]


def get_inference_model_extensions_filter():
    extensions = [
        "*.pt",
        "*.pth",  # PyTorch
        "*.onnx",  # ONNX
        "*.torchscript",
        "*.ptl",  # TorchScript
        "*.engine",  # TensorRT
        "*.tflite",  # TensorFlow Lite
        "*.pb",  # TensorFlow Frozen Graph
        "*.mlmodel",  # CoreML
        "*.xml",  # OpenVINO IR (main file)
        "*.param",  # NCNN (main file)
    ]
    return f"Model Files ({' '.join(extensions)});;All Files (*)"


def get_available_cameras():
    index = 0
    arr = []
    while index < 5:  # Check first 5 indices
        cap = cv2.VideoCapture(
            index, cv2.CAP_DSHOW
        )  # Use CAP_DSHOW for Windows for better performance/compatibility
        if cap.isOpened():
            if cap.read()[0]:  # Check if a frame can be read
                arr.append(index)
            cap.release()
        index += 1
    return arr


# --- Worker Threads ---


class InstallThread(QThread):  # For Ultralytics
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def run(self):
        try:
            self.log_message.emit("Installing ultralytics...")
            # Use a more robust way to capture output, especially errors
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "ultralytics"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.log_message.emit("Ultralytics installed successfully.")
                global ULTRALYTICS_INSTALLED
                ULTRALYTICS_INSTALLED = True
                self.finished.emit(True, "Installation successful!")
            else:
                error_msg = (
                    f"Installation failed: {stderr or stdout or 'Unknown pip error'}"
                )
                self.log_message.emit(error_msg)
                self.finished.emit(False, error_msg)

        except Exception as e:
            error_msg = f"An error occurred during installation: {e}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, error_msg)


class InstallSpecificDependencyThread(QThread):
    finished = pyqtSignal(
        bool, str, str
    )  # success, dependency_name, original_model_path_for_context
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
            "openvino": "openvino-dev",
            "tflite_runtime": "tflite-runtime",  # Common for tflite
            "tensorflow": "tensorflow",  # Fallback if tflite_runtime fails or for full TF
            "pycoral": "pycoral",
        }
        self.pip_package_name = dependency_map.get(self.dependency_name.lower())

        if not self.pip_package_name:
            # Try a more generic approach if a specific one isn't mapped (e.g. for some tf backends)
            if (
                "tensorflow" in self.dependency_name.lower()
                or "tflite" in self.dependency_name.lower()
            ):
                self.pip_package_name = "tensorflow"  # Default to full tensorflow if specific tflite_runtime is not the issue
                self.log_message.emit(
                    f"Specific package for {self.dependency_name} not found, trying generic 'tensorflow'."
                )
            else:
                self.log_message.emit(
                    f"Don't know how to auto-install: {self.dependency_name}. Please install manually."
                )
                self.finished.emit(False, self.dependency_name, self.model_path_context)
                return

        try:
            self.log_message.emit(
                f"Attempting to install {self.dependency_name} (pip: {self.pip_package_name})..."
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", self.pip_package_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.log_message.emit(
                    f"{self.dependency_name} (as {self.pip_package_name}) installed successfully via pip."
                )
                # Verification step
                verification_cmd = None
                if self.dependency_name.startswith("onnxruntime"):
                    verification_cmd = ["-c", "import onnxruntime"]
                elif self.dependency_name == "openvino":
                    verification_cmd = ["-c", "from openvino.runtime import Core"]
                elif (
                    "tflite" in self.dependency_name.lower()
                    or "tensorflow" in self.dependency_name.lower()
                ):
                    verification_cmd = [
                        "-c",
                        "import tensorflow as tf; print(tf.__version__)",
                    ]

                if verification_cmd:
                    subprocess.run(
                        [sys.executable] + verification_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        creationflags=(
                            subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32"
                            else 0
                        ),
                    )
                    self.log_message.emit(
                        f"Successfully verified import of {self.dependency_name}."
                    )

                self.finished.emit(True, self.dependency_name, self.model_path_context)
            else:
                error_msg = f"Installation of {self.dependency_name} (as {self.pip_package_name}) failed: {stderr or stdout or 'Unknown pip error'}"
                self.log_message.emit(error_msg)
                self.finished.emit(False, self.dependency_name, self.model_path_context)
        except subprocess.CalledProcessError as e:
            error_msg = f"Verification of {self.dependency_name} failed after pip install: {e.stderr or e.stdout or str(e)}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, self.dependency_name, self.model_path_context)
        except Exception as e:
            error_msg = f"An error occurred during {self.dependency_name} installation/verification: {e}"
            self.log_message.emit(error_msg)
            self.finished.emit(False, self.dependency_name, self.model_path_context)


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
            self.log_message.emit(
                f"Starting data preparation for {os.path.basename(self.zip_path)}..."
            )
            if os.path.exists(self.custom_data_dir):
                shutil.rmtree(self.custom_data_dir)
            if os.path.exists(self.final_data_dir):
                shutil.rmtree(self.final_data_dir)
            os.makedirs(self.custom_data_dir, exist_ok=True)
            os.makedirs(self.final_data_dir, exist_ok=True)
            self.progress.emit("Unzipping dataset...", 10)

            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                zip_ref.extractall(self.custom_data_dir)
            self.log_message.emit("Dataset unzipped.")
            self.progress.emit("Dataset unzipped.", 30)

            images_src, labels_src, classes_file_path = None, None, None

            possible_classes_files = glob.glob(
                os.path.join(self.custom_data_dir, "**", "classes.txt"), recursive=True
            )
            if possible_classes_files:
                classes_file_path = possible_classes_files[0]
                self.log_message.emit(f"Found classes.txt at {classes_file_path}")

            for i in range(3):
                search_path = self.custom_data_dir
                for _ in range(i):
                    dirs_in_search_path = [
                        d
                        for d in os.listdir(search_path)
                        if os.path.isdir(os.path.join(search_path, d))
                    ]
                    if len(dirs_in_search_path) == 1:
                        search_path = os.path.join(search_path, dirs_in_search_path[0])
                    else:
                        break

                possible_images_dirs = glob.glob(
                    os.path.join(search_path, "**", "images"), recursive=True
                )
                possible_labels_dirs = glob.glob(
                    os.path.join(search_path, "**", "labels"), recursive=True
                )

                if possible_images_dirs:
                    images_src = possible_images_dirs[0]
                if possible_labels_dirs:
                    labels_src = possible_labels_dirs[0]

                if images_src and labels_src:
                    self.log_message.emit(f"Found images in: {images_src}")
                    self.log_message.emit(f"Found labels in: {labels_src}")
                    break

            if not images_src or not os.path.isdir(images_src):
                self.finished.emit(
                    False,
                    "Could not find 'images' folder. Please ensure it's named 'images' and is within the zip structure.",
                )
                return
            if not labels_src or not os.path.isdir(labels_src):
                self.finished.emit(
                    False,
                    "Could not find 'labels' folder. Please ensure it's named 'labels' and is within the zip structure.",
                )
                return
            if not classes_file_path or not os.path.isfile(classes_file_path):
                self.log_message.emit(
                    "classes.txt not found. It must be present in the zip (e.g., next to 'images' and 'labels' folders or in their parent)."
                )
                self.finished.emit(
                    False,
                    "Could not find 'classes.txt'. It must be present in the zip.",
                )
                return

            self.progress.emit("Splitting data...", 50)
            all_images = sorted(
                [
                    f
                    for f in os.listdir(images_src)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
                ]
            )

            valid_images = []
            for img_file in all_images:
                base_name = os.path.splitext(img_file)[0]
                lbl_file = base_name + ".txt"
                if os.path.exists(os.path.join(labels_src, lbl_file)):
                    valid_images.append(img_file)
                else:
                    self.log_message.emit(
                        f"Warning: Label file for {img_file} not found. Skipping this image."
                    )

            if not valid_images:
                self.finished.emit(False, "No valid image-label pairs found.")
                return

            random.shuffle(valid_images)
            split_idx = int(len(valid_images) * self.train_pct)
            train_images = valid_images[:split_idx]
            val_images = valid_images[split_idx:]

            train_img_path = os.path.join(self.final_data_dir, "train", "images")
            train_lbl_path = os.path.join(self.final_data_dir, "train", "labels")
            val_img_path = os.path.join(self.final_data_dir, "val", "images")
            val_lbl_path = os.path.join(self.final_data_dir, "val", "labels")

            os.makedirs(train_img_path, exist_ok=True)
            os.makedirs(train_lbl_path, exist_ok=True)
            os.makedirs(val_img_path, exist_ok=True)
            os.makedirs(val_lbl_path, exist_ok=True)
            self.progress.emit("Copying files...", 70)

            def copy_files(
                file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir
            ):
                for img_file in file_list:
                    base_name = os.path.splitext(img_file)[0]
                    lbl_file = base_name + ".txt"
                    shutil.copy(
                        os.path.join(src_img_dir, img_file),
                        os.path.join(dst_img_dir, img_file),
                    )
                    shutil.copy(
                        os.path.join(src_lbl_dir, lbl_file),
                        os.path.join(dst_lbl_dir, lbl_file),
                    )

            copy_files(
                train_images, images_src, labels_src, train_img_path, train_lbl_path
            )
            copy_files(val_images, images_src, labels_src, val_img_path, val_lbl_path)
            self.log_message.emit("Train/validation files copied.")
            self.progress.emit("Files copied.", 90)

            with open(classes_file_path, "r", encoding="utf-8") as f:  # Added encoding
                classes = [line.strip() for line in f if line.strip()]

            data_yaml_content = {
                "path": os.path.abspath(self.final_data_dir),
                "train": os.path.join("train", "images"),
                "val": os.path.join("val", "images"),
                "nc": len(classes),
                "names": classes,
            }
            with open(self.yaml_path, "w", encoding="utf-8") as f:  # Added encoding
                yaml.dump(
                    data_yaml_content,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                )  # allow_unicode for names
            self.log_message.emit(f"data.yaml created at {self.yaml_path}")
            self.progress.emit("data.yaml created.", 100)
            self.finished.emit(True, self.yaml_path)

        except Exception as e:
            self.log_message.emit(f"Data preparation error: {e}")
            self.finished.emit(False, f"Data preparation failed: {e}")


class TrainThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self, model_name, data_yaml, epochs, imgsz, device, project_name="yolo_gui_runs"
    ):
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
            self.progress_update.emit(
                f"Initializing YOLO model: {self.model_name} for training."
            )
            model = YOLO(self.model_name)
            self.progress_update.emit(
                f"Starting training on device: {self.device if self.device is not None else 'cpu'}"
            )
            self.progress_update.emit(
                f"Dataset: {self.data_yaml}, Epochs: {self.epochs}, Image Size: {self.imgsz}"
            )

            task_type = (
                "detect" if "seg" not in self.model_name.lower() else "segment"
            )  # Check model_name for task
            exact_run_dir = os.path.join(self.project_name, task_type, self.run_name)
            if os.path.exists(exact_run_dir):
                self.progress_update.emit(
                    f"Deleting existing run directory: {exact_run_dir} to ensure a fresh start."
                )
                shutil.rmtree(exact_run_dir)

            device_arg = self.device if self.device not in ["cpu", None] else "cpu"
            if "cuda" in device_arg:
                device_arg = device_arg.split(":")[-1]

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                device=device_arg,
                project=self.project_name,
                name=self.run_name,
                exist_ok=False,
            )

            latest_train_dir = os.path.join(self.project_name, task_type, self.run_name)

            if not os.path.exists(latest_train_dir):
                train_dirs = glob.glob(
                    os.path.join(self.project_name, task_type, self.run_name + "*")
                )
                if not train_dirs:
                    self.finished.emit(
                        False, "Could not find training results directory."
                    )
                    return
                latest_train_dir = max(train_dirs, key=os.path.getctime)

            best_pt_path = os.path.join(latest_train_dir, "weights", "best.pt")

            if os.path.exists(best_pt_path):
                output_model_dir = "trained_models_gui"
                os.makedirs(output_model_dir, exist_ok=True)

                base_model_name = os.path.splitext(os.path.basename(self.model_name))[0]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_model_name = f"{base_model_name}_custom_{timestamp}.pt"
                final_model_path = os.path.join(output_model_dir, final_model_name)

                shutil.copy(best_pt_path, final_model_path)
                self.progress_update.emit(
                    f"Training finished. Best model saved as {final_model_path}"
                )
                self.finished.emit(True, os.path.abspath(final_model_path))
            else:
                self.progress_update.emit(
                    f"Training completed, but 'best.pt' not found in {latest_train_dir}/weights."
                )
                self.finished.emit(
                    False,
                    f"Training completed, but 'best.pt' not found in {latest_train_dir}/weights.",
                )

        except Exception as e:
            self.progress_update.emit(f"Training error: {e}")
            self.finished.emit(False, f"Training failed: {e}")


class InferenceThread(QThread):
    frame_ready = pyqtSignal(QImage, float, int)
    finished = pyqtSignal(str, str)
    log_message = pyqtSignal(str)

    def __init__(
        self,
        model_path,
        source_type,
        source_path,
        device,
        confidence_threshold=0.25,
        resolution=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.source_type = source_type
        self.source_path = source_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self._is_running = True
        self.model = None
        self.cap = None
        self.prev_time = 0
        self.fps = 0

    def run(self):
        try:
            self.log_message.emit(
                f"Attempting to load model for inference: {self.model_path} on device: {self.device}"
            )

            device_arg = self.device
            if "cuda" in device_arg:
                device_arg = device_arg.split(":")[-1]

            try:
                self.model = YOLO(self.model_path)
                if (
                    self.source_type != "image"
                ):  # Test with a dummy frame for stream sources
                    dummy_h = (
                        self.resolution[1]
                        if self.resolution and len(self.resolution) == 2
                        else 480
                    )
                    dummy_w = (
                        self.resolution[0]
                        if self.resolution and len(self.resolution) == 2
                        else 640
                    )
                    dummy_frame = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
                    self.model.predict(
                        dummy_frame,
                        device=device_arg,
                        conf=self.confidence_threshold,
                        verbose=False,
                        imgsz=(dummy_h, dummy_w),
                    )
                self.log_message.emit(
                    f"Model '{os.path.basename(self.model_path)}' loaded and test prediction successful."
                )

            except ImportError as e:
                err_str = str(e).lower()
                if "onnxruntime" in err_str:
                    self.log_message.emit(
                        f"ImportError: ONNX Runtime is likely missing for {self.model_path}."
                    )
                    self.finished.emit(
                        f"missing_dependency:onnxruntime:{self.model_path}",
                        self.source_type,
                    )
                    return
                # Add more specific checks for other backends if needed
                elif (
                    "tensorflow" in err_str or "tflite" in err_str
                ):  # Broader check for TF related
                    self.log_message.emit(
                        f"ImportError: TensorFlow/TFLite runtime is likely missing for {self.model_path}."
                    )
                    self.finished.emit(
                        f"missing_dependency:tensorflow:{self.model_path}",
                        self.source_type,
                    )  # Use generic 'tensorflow'
                    return
                self.log_message.emit(f"Unhandled ImportError during model load: {e}")
                self.finished.emit(
                    f"Error: Missing library for model - {e}", self.source_type
                )
                return
            except Exception as e:
                err_str = str(e).lower()
                if "onnxruntime" in err_str and (
                    "not found" in err_str or "install" in err_str
                ):
                    self.log_message.emit(
                        f"Error suggests ONNX Runtime is missing for {self.model_path}: {e}"
                    )
                    self.finished.emit(
                        f"missing_dependency:onnxruntime:{self.model_path}",
                        self.source_type,
                    )
                    return
                if "openvino" in err_str and (
                    "not found" in err_str
                    or "install" in err_str
                    or "inference engine" in err_str
                ):
                    self.log_message.emit(
                        f"Error suggests OpenVINO is missing or not configured for {self.model_path}: {e}"
                    )
                    self.finished.emit(
                        f"missing_dependency:openvino:{self.model_path}",
                        self.source_type,
                    )
                    return
                if ("tensorflow" in err_str or "tflite" in err_str) and (
                    "not found" in err_str or "install" in err_str
                ):
                    self.log_message.emit(
                        f"Error suggests TensorFlow/TFLite is missing for {self.model_path}: {e}"
                    )
                    self.finished.emit(
                        f"missing_dependency:tensorflow:{self.model_path}",
                        self.source_type,
                    )
                    return
                self.log_message.emit(
                    f"Error during YOLO model initialization or test prediction: {e}"
                )
                self.finished.emit(
                    f"Error: Could not load model or initial predict failed - {e}",
                    self.source_type,
                )
                return

            self.log_message.emit("Model loaded. Proceeding with inference setup.")

            if self.source_type == "webcam":
                try:
                    cam_idx = int(self.source_path)
                    self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.finished.emit(
                            f"Error: Could not open webcam {cam_idx}.", self.source_type
                        )
                        return
                    if self.resolution and len(self.resolution) == 2:
                        self.log_message.emit(
                            f"Setting webcam resolution to {self.resolution[0]}x{self.resolution[1]}"
                        )
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        self.log_message.emit(
                            f"Actual webcam resolution set to: {int(actual_width)}x{int(actual_height)}"
                        )
                        if not (
                            abs(actual_width - self.resolution[0]) < 10
                            and abs(actual_height - self.resolution[1]) < 10
                        ):  # Allow small tolerance
                            self.log_message.emit(
                                f"Warning: Webcam might not have accepted exact resolution {self.resolution}. Using {actual_width}x{actual_height}."
                            )
                    self.log_message.emit(f"Webcam {cam_idx} opened.")
                except ValueError:
                    self.finished.emit(
                        f"Error: Invalid webcam index '{self.source_path}'.",
                        self.source_type,
                    )
                    return
            elif self.source_type == "video":
                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    self.finished.emit(
                        f"Error: Could not open video file: {self.source_path}",
                        self.source_type,
                    )
                    return
                self.log_message.emit(f"Video file opened: {self.source_path}")
            elif self.source_type == "image":
                img = cv2.imread(self.source_path)
                if img is None:
                    self.finished.emit(
                        f"Error: Could not open image file: {self.source_path}",
                        self.source_type,
                    )
                    return
                self.log_message.emit(
                    f"Image file opened: {self.source_path}. Running prediction..."
                )

                results = self.model.predict(
                    img,
                    device=device_arg,
                    conf=self.confidence_threshold,
                    verbose=False,
                )

                annotated_frame = img.copy()
                object_count = 0
                if results and hasattr(results[0], "plot"):
                    try:
                        annotated_frame = results[0].plot()
                    except Exception as plot_err:
                        self.log_message.emit(
                            f"Warning: result.plot() failed: {plot_err}. Displaying original image."
                        )
                else:
                    self.log_message.emit(
                        "Warning: result.plot() not available or no results. Displaying original image."
                    )

                if (
                    results
                    and hasattr(results[0], "boxes")
                    and results[0].boxes is not None
                ):
                    object_count = len(results[0].boxes)

                self.emit_frame(annotated_frame, 0, object_count)
                self.finished.emit("Image inference complete.", self.source_type)
                return
            else:
                self.finished.emit(
                    "Error: Invalid source type for inference.", self.source_type
                )
                return

            self.prev_time = time.time()
            while self._is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message.emit("End of stream or cannot read frame.")
                    break

                current_time = time.time()
                elapsed_time = current_time - self.prev_time
                self.fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                self.prev_time = current_time

                results = self.model.predict(
                    frame,
                    device=device_arg,
                    conf=self.confidence_threshold,
                    verbose=False,
                )

                annotated_frame_loop = frame.copy()
                object_count_loop = 0
                if results and hasattr(results[0], "plot"):
                    try:
                        annotated_frame_loop = results[0].plot()
                    except Exception as plot_err_loop:
                        self.log_message.emit(
                            f"Warning: result.plot() failed in loop: {plot_err_loop}."
                        )
                if (
                    results
                    and hasattr(results[0], "boxes")
                    and results[0].boxes is not None
                ):
                    object_count_loop = len(results[0].boxes)

                self.emit_frame(annotated_frame_loop, self.fps, object_count_loop)
                QThread.msleep(1)

            if self.cap:
                self.cap.release()
            self.finished.emit("Inference stopped.", self.source_type)

        except ImportError as e:
            err_str = str(e).lower()
            if "onnxruntime" in err_str:
                self.log_message.emit(
                    f"Delayed ImportError: ONNX Runtime is likely missing for {self.model_path}."
                )
                self.finished.emit(
                    f"missing_dependency:onnxruntime:{self.model_path}",
                    self.source_type,
                )
            elif "tensorflow" in err_str or "tflite" in err_str:
                self.log_message.emit(
                    f"Delayed ImportError: TensorFlow/TFLite runtime is likely missing for {self.model_path}."
                )
                self.finished.emit(
                    f"missing_dependency:tensorflow:{self.model_path}", self.source_type
                )
            else:
                self.log_message.emit(
                    f"Unhandled Delayed ImportError during inference: {e}"
                )
                self.finished.emit(
                    f"Error: Missing library during inference - {e}", self.source_type
                )
        except Exception as e:
            self.log_message.emit(f"General inference error: {e}")
            err_str_runtime = str(e).lower()
            if "onnxruntime" in err_str_runtime and (
                "not found" in err_str_runtime or "install" in err_str_runtime
            ):
                self.log_message.emit(
                    f"Runtime Error suggests ONNX Runtime is missing: {e}"
                )
                self.finished.emit(
                    f"missing_dependency:onnxruntime:{self.model_path}",
                    self.source_type,
                )
            elif "openvino" in err_str_runtime and (
                "not found" in err_str_runtime
                or "install" in err_str_runtime
                or "inference engine" in err_str_runtime
            ):
                self.log_message.emit(
                    f"Runtime Error suggests OpenVINO is missing or not configured: {e}"
                )
                self.finished.emit(
                    f"missing_dependency:openvino:{self.model_path}", self.source_type
                )
            elif ("tensorflow" in err_str_runtime or "tflite" in err_str_runtime) and (
                "not found" in err_str_runtime or "install" in err_str_runtime
            ):
                self.log_message.emit(
                    f"Runtime Error suggests TensorFlow/TFLite is missing: {e}"
                )
                self.finished.emit(
                    f"missing_dependency:tensorflow:{self.model_path}", self.source_type
                )
            else:
                self.finished.emit(f"Inference error: {e}", self.source_type)
        finally:
            if self.cap:
                self.cap.release()
            self.log_message.emit("Inference thread resources potentially released.")

    def emit_frame(self, frame_cv, fps, obj_count):
        if frame_cv is None:
            self.log_message.emit("emit_frame called with None frame.")
            return
        try:
            if len(frame_cv.shape) == 2:
                frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_GRAY2BGR)

            rgb_image = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

            scaled_image = qt_image.scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.frame_ready.emit(scaled_image, fps, obj_count)
        except cv2.error as e:
            self.log_message.emit(
                f"cv2 error in emit_frame: {e}. Frame shape: {frame_cv.shape if frame_cv is not None else 'None'}"
            )
        except Exception as e:
            self.log_message.emit(f"Error in emit_frame: {e}")

    def stop(self):
        self.log_message.emit("Stopping inference thread...")
        self._is_running = False


class ExportThread(QThread):
    finished = pyqtSignal(bool, str)
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

            output_dir = "exported_models_gui"
            os.makedirs(output_dir, exist_ok=True)

            exported_path_or_dir = model.export(format=self.export_format, imgsz=640)

            final_destination = exported_path_or_dir

            if isinstance(exported_path_or_dir, str) and os.path.exists(
                exported_path_or_dir
            ):
                base_exported_name = os.path.basename(exported_path_or_dir)
                target_path = os.path.join(output_dir, base_exported_name)

                if os.path.normpath(exported_path_or_dir) != os.path.normpath(
                    target_path
                ):
                    if os.path.exists(target_path):
                        if os.path.isdir(target_path):
                            shutil.rmtree(target_path)
                        else:
                            os.remove(target_path)

                    shutil.move(exported_path_or_dir, target_path)
                    final_destination = target_path
                else:
                    final_destination = exported_path_or_dir

            final_destination = os.path.abspath(final_destination)
            self.log_message.emit(
                f"Model successfully exported to: {final_destination}"
            )
            self.finished.emit(True, f"Model exported to {final_destination}")
        except Exception as e:
            self.log_message.emit(f"Export failed: {e}")
            self.finished.emit(False, f"Export failed: {e}")


# Add this after the ExportThread class and before the ModernYoloGUI class
class ModelDownloadThread(QThread):
    progress = pyqtSignal(str, int)  # Changed to include progress percentage
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, model_name, target_path):
        super().__init__()
        self.model_name = model_name
        self.target_path = target_path
        self._is_running = True

    def run(self):
        try:
            self.log_message.emit(f"Starting download of {self.model_name}...")
            from ultralytics import YOLO
            from tqdm import tqdm
            import requests
            import os

            # Get the model URL
            model = YOLO(self.model_name)
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{self.model_name}"
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            self.log_message.emit(f"Downloading {self.model_name} ({total_size/1024/1024:.1f} MB)...")
            
            with open(self.target_path, 'wb') as file:
                downloaded = 0
                for data in response.iter_content(block_size):
                    if not self._is_running:
                        self.log_message.emit("Download cancelled")
                        return
                    file.write(data)
                    downloaded += len(data)
                    progress = int((downloaded / total_size) * 100)
                    self.progress.emit(f"Downloading {self.model_name}...", progress)
            
            self.log_message.emit(f"✓ Model {self.model_name} downloaded successfully to {self.target_path}")
            self.finished.emit(True, self.target_path)
        except Exception as e:
            self.log_message.emit(f"✗ Error downloading model: {e}")
            self.finished.emit(False, str(e))

    def stop(self):
        self._is_running = False


# --- Main Application Window ---
class ModernYoloGUI(QMainWindow):
    APP_VERSION = "0.1.0"  # Incremented version

    def __init__(self):
        super().__init__()

        # Define palettes first as they are used by get_icon indirectly via theme
        self.light_palette = {
            "window_bg": "#F0F0F0",
            "nav_bar_bg": "#E8E8E8",
            "nav_button_bg": "transparent",
            "nav_button_hover_bg": "#D8D8D8",
            "nav_button_checked_bg": "#C8C8C8",
            "nav_button_text": "#333333",
            "text_color": "#222222",
            "label_header_color": "#007AFF",
            "border_color": "#CCCCCC",
            "group_bg": "#FFFFFF",
            "button_bg": "#E0E0E0",
            "button_hover_bg": "#D0D0D0",
            "button_pressed_bg": "#B0B0B0",
            "button_text": "#333333",
            "input_bg": "#FFFFFF",
            "input_border": "#BDBDBD",
            "disabled_text": "#AAAAAA",
            "disabled_bg": "#E0E0E0",
            "scroll_bar_bg": "#F0F0F0",
            "scroll_bar_handle": "#C0C0C0",
            "log_bg": "#FDFDFD",
            "log_text": "#333333",
            "video_feed_border": "#CCCCCC",
            "video_feed_bg": "#DDDDDD",
            "icon_button_bg": "transparent",
            "icon_button_hover_bg": "#D0D0D0",
            "icon_button_pressed_bg": "#B0B0B0",
        }
        self.dracula_palette = {
            "window_bg": "#282a36",
            "nav_bar_bg": "#1e1f28",
            "nav_button_bg": "transparent",
            "nav_button_hover_bg": "#44475a",
            "nav_button_checked_bg": "#6272a4",
            "nav_button_text": "#f8f8f2",
            "text_color": "#f8f8f2",
            "label_header_color": "#8be9fd",
            "border_color": "#44475a",
            "group_bg": "#353746",
            "button_bg": "#44475a",
            "button_hover_bg": "#6272a4",
            "button_pressed_bg": "#707EAA",
            "button_text": "#f8f8f2",
            "input_bg": "#282a36",
            "input_border": "#6272a4",
            "disabled_text": "#6272a4",
            "disabled_bg": "#343746",
            "scroll_bar_bg": "#282a36",
            "scroll_bar_handle": "#44475a",
            "log_bg": "#21222c",
            "log_text": "#bd93f9",
            "video_feed_border": "#6272a4",
            "video_feed_bg": "#1e1f28",
            "icon_button_bg": "transparent",
            "icon_button_hover_bg": "#6272a4",
            "icon_button_pressed_bg": "#44475a",
        }
        self.current_theme = "dark"  # Changed from "light" to "dark"
        self.current_palette_colors = self.dracula_palette  # Initialize with dark theme

        self.setWindowTitle(f"VisionCraft Studio v{self.APP_VERSION}")
        self.setGeometry(50, 50, 1000, 800)
        self.setWindowIcon(self.get_icon("app_icon", fallback_text=False))

        self.dataset_zip_path = None
        self.data_yaml_path = None
        self.trained_model_path_for_inference = None
        self.inference_thread = None
        self.dependency_install_thread = None

        self.work_dir = "yolo_gui_workspace"
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "dataset_processing"), exist_ok=True)
        os.makedirs("trained_models_gui", exist_ok=True)
        os.makedirs("exported_models_gui", exist_ok=True)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.stacked_widget = QStackedWidget()

        self.home_page = self.create_home_page()
        self.data_page = self.create_data_page()
        self.train_page = self.create_train_page()
        self.deploy_page = self.create_deploy_page()
        self.settings_page = self.create_settings_page()
        self.about_page = self.create_about_page()

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.data_page)
        self.stacked_widget.addWidget(self.train_page)
        self.stacked_widget.addWidget(self.deploy_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.about_page)

        self.nav_bar = self.create_navigation_bar()

        self.log_area_widget = QWidget()
        self.log_area_widget.setObjectName("logAreaWidget")
        self.log_area_layout = QHBoxLayout(self.log_area_widget)
        self.log_area_layout.setContentsMargins(10, 5, 10, 5)
        self.log_area_layout.setSpacing(5)

        self.log_group = QGroupBox("Console Log")
        log_group_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)  # Increased minimum height
        self.log_text.setMaximumHeight(200)  # Added maximum height
        log_group_layout.addWidget(self.log_text)
        self.log_group.setLayout(log_group_layout)

        # Create a container for the console toggle
        toggle_container = QWidget()
        toggle_container.setObjectName("toggleContainer")
        toggle_layout = QHBoxLayout(toggle_container)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(5)

        self.toggle_console_button = QPushButton()
        self.toggle_console_button.setCheckable(True)
        self.toggle_console_button.setChecked(False)
        self.toggle_console_button.setToolTip("Show/Hide Console")
        self.toggle_console_button.clicked.connect(
            self.toggle_console_visibility_action
        )
        self.toggle_console_button.setObjectName("consoleToggle")
        self.toggle_console_button.setFixedSize(
            QSize(36, 20)
        )  # Apple-style toggle size
        toggle_layout.addWidget(self.toggle_console_button)

        self.log_area_layout.addWidget(self.log_group, 1)
        self.log_area_layout.addWidget(toggle_container, 0, Qt.AlignTop | Qt.AlignRight)

        self.main_layout.addWidget(self.stacked_widget)
        self.main_layout.addWidget(self.nav_bar)
        self.main_layout.addWidget(self.log_area_widget)

        self.apply_stylesheet(self.current_theme)

        self.log(f"VisionCraft Studio v{self.APP_VERSION} started.")
        self.update_ultralytics_status_display()
        if not ULTRALYTICS_INSTALLED:
            self.log(
                "Ultralytics library is not detected. Please install it via the Settings page."
            )

        self.update_inference_controls_state()
        self.video_feed_label.setText("Output Preview")
        self.toggle_console_visibility_action()

    def get_icon(self, name_no_ext, extension=".png", fallback_text=True):
        icon_path = os.path.join("icons", name_no_ext + extension)
        if os.path.exists(icon_path):
            return QIcon(icon_path)

        if not fallback_text:
            self.log(f"Critical Icon missing: icons/{name_no_ext}{extension}")
        return QIcon()

    def toggle_console_visibility_action(self):
        if self.toggle_console_button.isChecked():
            self.log_group.show()
            self.log_text.setMinimumHeight(150)  # Match the minimum height
            self.log_text.setMaximumHeight(200)  # Match the maximum height
            self.toggle_console_button.setIcon(self.get_icon("console_hide"))
            self.toggle_console_button.setToolTip("Hide Console")
            self.log("Console shown.")
        else:
            self.log_group.hide()
            self.toggle_console_button.setIcon(self.get_icon("console_show"))
            self.toggle_console_button.setToolTip("Show Console")
            print(f"[{time.strftime('%H:%M:%S')}] Console hidden.")

    def create_navigation_bar(self):
        nav_widget = QWidget()
        nav_widget.setObjectName("nav_widget")
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        icon_path_base = "icons"  # Relative path

        self.nav_buttons_data = [
            {"text": "Home", "icon_name": "home", "page_index": 0},
            {"text": "Data", "icon_name": "data", "page_index": 1},
            {"text": "Train", "icon_name": "train", "page_index": 2},
            {"text": "Deploy", "icon_name": "deploy", "page_index": 3},
            {"text": "Settings", "icon_name": "settings", "page_index": 4},
            {"text": "About", "icon_name": "about", "page_index": 5},
        ]

        self.nav_buttons = []
        for item in self.nav_buttons_data:
            icon_full_path = os.path.join(icon_path_base, item["icon_name"] + ".png")
            q_icon = (
                QIcon(icon_full_path) if os.path.exists(icon_full_path) else QIcon()
            )

            button = QPushButton(q_icon, f" {item['text']}")
            button.setIconSize(QSize(22, 22))
            button.setCheckable(True)
            button.setProperty("class", "navButton")
            button.clicked.connect(
                lambda checked, index=item["page_index"]: self.switch_page(index)
            )
            nav_layout.addWidget(button)
            self.nav_buttons.append(button)

        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)
        nav_widget.setFixedHeight(50)
        return nav_widget

    def update_all_icons(self):
        for i, item in enumerate(self.nav_buttons_data):
            if i < len(self.nav_buttons):  # Ensure button exists
                self.nav_buttons[i].setIcon(self.get_icon(item["icon_name"]))

        if self.toggle_console_button.isChecked():
            self.toggle_console_button.setIcon(self.get_icon("console_hide"))
        else:
            self.toggle_console_button.setIcon(self.get_icon("console_show"))

        if hasattr(self, "load_dataset_button"):
            self.load_dataset_button.setIcon(self.get_icon("dataset_load"))
        if hasattr(self, "train_button"):
            self.train_button.setIcon(self.get_icon("train_action"))
        if hasattr(self, "load_trained_model_button"):
            self.load_trained_model_button.setIcon(self.get_icon("model_load"))

        if hasattr(self, "start_webcam_button"):
            self.update_start_webcam_button_state()
        if hasattr(self, "load_media_button"):
            self.load_media_button.setIcon(self.get_icon("media_load"))
        if hasattr(self, "stop_inference_button"):
            self.stop_inference_button.setIcon(self.get_icon("stop_inference"))
        if hasattr(self, "export_model_button"):
            self.export_model_button.setIcon(self.get_icon("model_export"))
        if hasattr(self, "install_ultralytics_button"):
            self.install_ultralytics_button.setIcon(self.get_icon("yolo_icon"))

        self.setWindowIcon(self.get_icon("app_icon", fallback_text=False))

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        page_name = self.nav_buttons[index].text().strip()
        current_time_str = time.strftime("%H:%M:%S")
        if self.log_group.isVisible():
            self.log(f"Switched to {page_name} page.")
        else:
            print(f"[{current_time_str}] Switched to {page_name} page.")

    def create_page_widget(self):
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(15, 15, 15, 15)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("pageScrollArea")
        scroll_area.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)

        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        return page, content_layout

    def create_home_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("HomePage")

        # Create a container widget for the background image with fixed aspect ratio
        background_container = QWidget()
        background_container.setObjectName("backgroundContainer")
        background_container.setMinimumHeight(600)
        background_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        background_layout = QVBoxLayout(background_container)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(0)

        # Create a semi-transparent overlay for better text readability
        content_overlay = QWidget()
        content_overlay.setObjectName("contentOverlay")
        content_layout = QVBoxLayout(content_overlay)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(30)

        # Add a spacer at the top for better vertical centering
        content_layout.addSpacing(20)

        # Title with improved styling
        title = QLabel("Welcome to VisionCraft Studio")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title)

        # Create a container for the intro text with glass effect
        intro_container = QWidget()
        intro_container.setObjectName("introContainer")
        intro_layout = QVBoxLayout(intro_container)
        intro_layout.setContentsMargins(40, 40, 40, 40)
        intro_layout.setSpacing(25)

        intro_html = f"""
        <div style='text-align: center;'>
            <p style='font-size: 14pt; margin: 20px 0; line-height: 1.6; font-weight: 500; color: {self.current_palette_colors["text_color"]};'>
                Your integrated environment for YOLO model development (v{self.APP_VERSION})
            </p>
            <p style='font-size: 13pt; margin: 25px 0; line-height: 1.6; color: {self.current_palette_colors["label_header_color"]};'>
                Navigate using the tabs below:
            </p>
            <div style='text-align: left; margin: 30px auto; max-width: 600px;'>
                <ul style='font-size: 12pt; line-height: 2.0; list-style-type: none; padding: 0;'>
                    <li style='margin: 15px 0; padding: 15px; border-radius: 12px; color: #333333;'>
                        <b style='font-size: 13pt;'>Data</b><br>
                        <span style='font-size: 11pt; opacity: 0.9;'>Prepare and manage your datasets</span>
                    </li>
                    <li style='margin: 15px 0; padding: 15px; border-radius: 12px; color: #333333;'>
                        <b style='font-size: 13pt;'>Train</b><br>
                        <span style='font-size: 11pt; opacity: 0.9;'>Configure and run model training</span>
                    </li>
                    <li style='margin: 15px 0; padding: 15px; border-radius: 12px; color: #333333;'>
                        <b style='font-size: 13pt;'>Deploy</b><br>
                        <span style='font-size: 11pt; opacity: 0.9;'>Test models, run live inference, and export</span>
                    </li>
                    <li style='margin: 15px 0; padding: 15px; border-radius: 12px; color: #333333;'>
                        <b style='font-size: 13pt;'>Settings</b><br>
                        <span style='font-size: 11pt; opacity: 0.9;'>Manage configurations and dependencies</span>
                    </li>
                    <li style='margin: 15px 0; padding: 15px; border-radius: 12px; color: #333333;'>
                        <b style='font-size: 13pt;'>About</b><br>
                        <span style='font-size: 11pt; opacity: 0.9;'>Information about this application</span>
                    </li>
                </ul>
            </div>
            <p style='font-size: 12pt; margin: 25px 0; line-height: 1.6; color: {self.current_palette_colors["label_header_color"]};'>
                Ensure Ultralytics is installed via the Settings page for full functionality
            </p>
        </div>
        """
        self.intro_text_label = QLabel()
        self.intro_text_label.setTextFormat(Qt.RichText)
        self.intro_text_label.setText(intro_html)
        self.intro_text_label.setWordWrap(True)
        self.intro_text_label.setObjectName("introText")
        intro_layout.addWidget(self.intro_text_label)

        content_layout.addWidget(intro_container)

        # Stats group with improved styling
        stats_group = QGroupBox("Quick Overview")
        stats_group.setObjectName("statsGroup")
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(30, 30, 30, 30)
        stats_layout.setSpacing(40)

        # Create individual stat containers
        models_container = QWidget()
        models_container.setObjectName("statContainer")
        models_layout = QVBoxLayout(models_container)
        models_layout.setAlignment(Qt.AlignCenter)
        
        models_icon = QLabel()
        models_icon.setObjectName("statIcon")
        models_icon.setAlignment(Qt.AlignCenter)
        models_icon.setPixmap(self.get_icon("train").pixmap(QSize(32, 32)))
        models_layout.addWidget(models_icon)
        
        self.models_trained_label = QLabel("Models Trained: 0")
        self.models_trained_label.setObjectName("statsLabel")
        self.models_trained_label.setAlignment(Qt.AlignCenter)
        models_layout.addWidget(self.models_trained_label)

        datasets_container = QWidget()
        datasets_container.setObjectName("statContainer")
        datasets_layout = QVBoxLayout(datasets_container)
        datasets_layout.setAlignment(Qt.AlignCenter)
        
        datasets_icon = QLabel()
        datasets_icon.setObjectName("statIcon")
        datasets_icon.setAlignment(Qt.AlignCenter)
        datasets_icon.setPixmap(self.get_icon("data").pixmap(QSize(32, 32)))
        datasets_layout.addWidget(datasets_icon)
        
        self.datasets_prepared_label = QLabel("Datasets Prepared: 0")
        self.datasets_prepared_label.setObjectName("statsLabel")
        self.datasets_prepared_label.setAlignment(Qt.AlignCenter)
        datasets_layout.addWidget(self.datasets_prepared_label)

        stats_layout.addWidget(models_container)
        stats_layout.addWidget(datasets_container)
        stats_group.setLayout(stats_layout)
        content_layout.addWidget(stats_group)

        # Add a spacer at the bottom for better vertical centering
        content_layout.addSpacing(20)
        content_layout.addStretch()

        background_layout.addWidget(content_overlay)
        layout.addWidget(background_container)

        return page

    def create_data_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("DataPage")

        group = QGroupBox("Dataset Preparation")
        group.setObjectName("contentGroup")
        data_layout = QVBoxLayout()

        self.data_zip_label = QLabel("No dataset (.zip) selected.")
        self.data_zip_label.setWordWrap(True)
        self.load_dataset_button = QPushButton(
            self.get_icon("dataset_load"), " Load Custom Dataset (.zip)"
        )
        self.load_dataset_button.clicked.connect(self.select_dataset_zip)
        data_layout.addWidget(self.data_zip_label)
        data_layout.addWidget(self.load_dataset_button)

        self.data_prep_progress = QProgressBar()
        self.data_prep_status_label = QLabel("Status: Waiting for dataset...")
        data_layout.addWidget(self.data_prep_progress)
        data_layout.addWidget(self.data_prep_status_label)

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

        # Add download button with modern styling
        download_container = QWidget()
        download_container.setObjectName("downloadContainer")
        download_layout = QHBoxLayout(download_container)
        download_layout.setContentsMargins(0, 10, 0, 0)
        
        self.download_model_button = QPushButton(self.get_icon("model_download"), " Download Selected Model")
        self.download_model_button.setObjectName("downloadButton")
        self.download_model_button.setCursor(Qt.PointingHandCursor)
        self.download_model_button.clicked.connect(self.download_selected_model)
        
        # Add loading spinner (initially hidden)
        self.download_spinner = QLabel()
        self.download_spinner.setObjectName("downloadSpinner")
        self.download_spinner.setFixedSize(16, 16)
        self.download_spinner.hide()
        
        download_layout.addWidget(self.download_model_button)
        download_layout.addWidget(self.download_spinner)
        download_layout.addStretch()
        
        model_layout.addWidget(download_container)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        params_group = QGroupBox("2. Training Parameters")
        params_group.setObjectName("contentGroup")
        params_form_layout = QGridLayout()

        params_form_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(25)
        params_form_layout.addWidget(self.epochs_spinbox, 0, 1)

        params_form_layout.addWidget(QLabel("Image Size (px):"), 1, 0)
        self.imgsz_spinbox = QSpinBox()
        self.imgsz_spinbox.setRange(32, 2048)
        self.imgsz_spinbox.setValue(640)
        self.imgsz_spinbox.setSingleStep(32)
        params_form_layout.addWidget(self.imgsz_spinbox, 1, 1)

        params_form_layout.addWidget(QLabel("Training Run Project Name:"), 2, 0)
        self.train_project_name_input = QComboBox()
        self.train_project_name_input.setEditable(True)
        self.train_project_name_input.addItems(["yolo_gui_runs", "my_custom_project"])
        self.train_project_name_input.setCurrentText("yolo_gui_runs")
        params_form_layout.addWidget(self.train_project_name_input, 2, 1)
        params_group.setLayout(params_form_layout)
        layout.addWidget(params_group)

        action_group = QGroupBox("3. Start Training")
        action_group.setObjectName("contentGroup")
        action_layout = QVBoxLayout()
        self.train_button = QPushButton(
            self.get_icon("train_action"), " Start Training"
        )
        self.train_button.clicked.connect(self.start_training_process)
        self.train_button.setEnabled(False)
        action_layout.addWidget(self.train_button)

        self.train_status_label = QLabel("Status: Configure data and model first.")
        action_layout.addWidget(self.train_status_label)

        self.trained_model_path_display_label = QLabel(
            "Trained Model: Not yet trained."
        )
        self.trained_model_path_display_label.setWordWrap(True)
        action_layout.addWidget(self.trained_model_path_display_label)

        self.load_model_after_train_checkbox = QCheckBox(
            "Load this model in Deploy tab after training"
        )
        self.load_model_after_train_checkbox.setChecked(True)
        action_layout.addWidget(self.load_model_after_train_checkbox)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        layout.addStretch()
        return page

    def create_deploy_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("DeployPage")

        load_model_group = QGroupBox("1. Load Model for Inference/Export")
        load_model_group.setObjectName("contentGroup")
        load_model_layout = QVBoxLayout()
        self.inference_model_status_label = QLabel("No model loaded for deployment.")
        self.inference_model_status_label.setWordWrap(True)
        self.load_trained_model_button = QPushButton(
            self.get_icon("model_load"), " Load Model for Inference"
        )
        self.load_trained_model_button.clicked.connect(self.select_model_for_inference)
        load_model_layout.addWidget(self.inference_model_status_label)
        load_model_layout.addWidget(self.load_trained_model_button)
        load_model_group.setLayout(load_model_layout)
        layout.addWidget(load_model_group)

        inference_group = QGroupBox("2. Live Inference / Test")
        inference_group.setObjectName("contentGroup")
        inf_layout = QVBoxLayout()

        # Create a container widget for the video feed to maintain proper layout
        video_container = QWidget()
        video_container.setObjectName("videoContainer")
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setAlignment(Qt.AlignCenter)

        self.video_feed_label = QLabel()
        self.video_feed_label.setMinimumSize(640, 480)
        self.video_feed_label.setMaximumSize(640, 480)
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_feed_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        video_container_layout.addWidget(self.video_feed_label)

        inf_layout.addWidget(video_container)

        stats_display_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("infoLabel")
        self.detected_objects_label = QLabel("Objects: --")
        self.detected_objects_label.setObjectName("infoLabel")
        stats_display_layout.addWidget(self.fps_label)
        stats_display_layout.addStretch()
        stats_display_layout.addWidget(self.detected_objects_label)
        inf_layout.addLayout(stats_display_layout)

        inf_controls_layout = QGridLayout()

        inf_controls_layout.addWidget(QLabel("Webcam:"), 0, 0)
        self.webcam_combo = QComboBox()
        self.update_webcam_list()
        inf_controls_layout.addWidget(self.webcam_combo, 0, 1)

        self.start_webcam_button = QPushButton()
        self.start_webcam_button.clicked.connect(self.toggle_webcam_inference_action)
        inf_controls_layout.addWidget(self.start_webcam_button, 0, 2, 2, 1)

        inf_controls_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.webcam_resolution_combo = QComboBox()
        self.webcam_resolutions = {
            "Default": None,
            "320x240": (320, 240),
            "640x480": (640, 480),
            "800x600": (800, 600),
            "1280x720 (HD)": (1280, 720),
            "1920x1080 (FHD)": (1920, 1080),
        }
        self.webcam_resolution_combo.addItems(self.webcam_resolutions.keys())
        self.webcam_resolution_combo.setCurrentText("640x480")
        inf_controls_layout.addWidget(self.webcam_resolution_combo, 1, 1)

        self.load_media_button = QPushButton(
            self.get_icon("media_load"), " Load Video/Image File"
        )
        self.load_media_button.clicked.connect(self.run_file_inference)
        inf_controls_layout.addWidget(self.load_media_button, 2, 0, 1, 2)

        self.stop_inference_button = QPushButton(
            self.get_icon("stop_inference"), " Stop File Inference"
        )
        self.stop_inference_button.setToolTip(
            "Stops inference from video/image file if running."
        )
        self.stop_inference_button.clicked.connect(self.stop_live_inference)
        inf_controls_layout.addWidget(self.stop_inference_button, 2, 2)

        inf_layout.addLayout(inf_controls_layout)
        inference_group.setLayout(inf_layout)
        layout.addWidget(inference_group)

        export_group = QGroupBox("3. Model Conversion (Export)")
        export_group.setObjectName("contentGroup")
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Export to Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(get_export_formats())
        self.export_model_button = QPushButton(
            self.get_icon("model_export"), " Convert & Export Model"
        )
        self.export_model_button.clicked.connect(self.run_export_model)

        exp_layout.addWidget(self.export_format_combo)
        exp_layout.addWidget(self.export_model_button)
        export_group.setLayout(exp_layout)
        layout.addWidget(export_group)
        self.export_status_label = QLabel("Export Status: Ready")
        layout.addWidget(self.export_status_label)

        layout.addStretch()
        self.update_inference_controls_state()
        return page

    def create_settings_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("SettingsPage")

        ultralytics_group = QGroupBox("Ultralytics Dependency Management")
        ultralytics_group.setObjectName("contentGroup")
        ult_layout = QVBoxLayout()
        self.ultralytics_status_label = QLabel("Ultralytics Status: Unknown")
        self.install_ultralytics_button = QPushButton(
            self.get_icon("yolo_icon"), " Install/Verify Ultralytics"
        )
        self.install_ultralytics_button.clicked.connect(
            self.manage_ultralytics_installation
        )
        ult_layout.addWidget(self.ultralytics_status_label)
        ult_layout.addWidget(self.install_ultralytics_button)
        ultralytics_group.setLayout(ult_layout)
        layout.addWidget(ultralytics_group)

        device_group = QGroupBox("Compute Device Configuration")
        device_group.setObjectName("contentGroup")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Select Default Compute Device:"))
        self.device_combo = QComboBox()
        self.populate_device_combo()
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        theme_group = QGroupBox("Appearance")
        theme_group.setObjectName("contentGroup")
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("UI Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark (Dracula)"])
        self.theme_combo.setCurrentText("Dark (Dracula)")  # Set dark theme as default
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        layout.addStretch()
        return page

    def create_about_page(self):
        page, layout = self.create_page_widget()
        page.setObjectName("AboutPage")

        # Create a container for better styling
        content_container = QWidget()
        content_container.setObjectName("aboutContentContainer")
        content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Add this line
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        content_layout.setAlignment(Qt.AlignTop)  # Add this line

        # Title with improved styling
        title_label = QLabel("VisionCraft Studio")
        title_label.setObjectName("pageTitle")
        title_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(title_label)

        # Version with subtle styling
        version_label = QLabel(f"Version {self.APP_VERSION}")
        version_label.setObjectName("versionLabel")
        version_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(version_label)

        content_layout.addSpacing(20)

        # Made with love section with improved styling
        heart_color = self.current_palette_colors.get("label_header_color", "#007AFF")
        made_with_html = f"""
        <div style='text-align: center;'>
            <p style='font-size: 13pt; margin: 15px 0; line-height: 1.4;'>
                Made with <span style='color: {heart_color}; font-size: 16pt; vertical-align: middle;'>&hearts;</span> 
                <span style='vertical-align: middle;'>by Midhun Mathew</span>
            </p>
        </div>
        """
        made_with_label = QLabel(made_with_html)
        made_with_label.setTextFormat(Qt.RichText)
        made_with_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(made_with_label)

        content_layout.addSpacing(20)

        # Description with improved styling
        description_html = f"""
        <div style='text-align: center; padding: 0 20px;'>
            <p style='font-size: 11pt; line-height: 1.6; margin: 10px 0;'>
                VisionCraft Studio is an integrated environment designed to simplify 
                the workflow of training and deploying YOLO object detection and segmentation models.
            </p>
            <p style='font-size: 11pt; line-height: 1.6; margin: 10px 0;'>
                It provides tools for data preparation, model training, live inference testing, 
                and model exporting.
            </p>
        </div>
        """
        description_label = QLabel(description_html)
        description_label.setTextFormat(Qt.RichText)
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(description_label)

        content_layout.addSpacing(30)

        # Links section with improved styling
        link_color = self.current_palette_colors.get("label_header_color", "#007AFF")
        support_link_html = f"""
        <div style='text-align: center;'>
            <p style='margin: 15px 0;'>
                <a href='https://coff.ee/memidhun' 
                   style='color: {link_color}; text-decoration: none; font-size: 11pt;'>
                    Support the Developer (Buy Me a Coffee)
                </a>
            </p>
            <p style='margin: 15px 0;'>
                <a href='https://www.linkedin.com/in/midhunmathew2002/' 
                   style='color: {link_color}; text-decoration: none; font-size: 11pt;'>
                    Connect on LinkedIn
                </a>
            </p>
        </div>
        """
        support_label = QLabel(support_link_html)
        support_label.setTextFormat(Qt.RichText)
        support_label.setOpenExternalLinks(True)
        support_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(support_label)

        # Add the content container to the main layout with stretch
        layout.addWidget(content_container, 1)  # Add stretch factor of 1
        layout.addStretch()

        return page

    def apply_stylesheet(self, theme="light"):
        self.current_theme = theme.lower().split(" ")[0]

        font = QFont("SF Pro Display", 10)
        if sys.platform == "win32":
            font.setFamily("Segoe UI")
        elif sys.platform == "darwin":
            font.setFamily("SF Pro Display")
        else:
            font.setFamily("Cantarell")
        QApplication.setFont(font)

        p = (
            self.light_palette
            if self.current_theme == "light"
            else self.dracula_palette
        )
        self.current_palette_colors = p

        # Add background image styling with theme-specific images
        light_bg_path = os.path.abspath(os.path.join("icons", "light_theme_bg.jpg"))
        dark_bg_path = os.path.abspath(os.path.join("icons", "dark_theme_bg.jpg"))

        # Select the appropriate background image based on theme
        bg_path = light_bg_path if self.current_theme == "light" else dark_bg_path

        if os.path.exists(bg_path):
            # Convert Windows path to URL format for Qt
            background_image_url = bg_path.replace("\\", "/")
            background_style = f"""
                /* Ensure the main container, the page itself, and the content overlay are transparent */
                QWidget#backgroundContainer, QWidget#HomePage, QWidget#contentOverlay {{ background-color: transparent; }}

                QWidget#backgroundContainer {{
                    background-image: url({background_image_url});
                    background-position: center;
                    background-repeat: no-repeat;
                    background-size: cover;
                    min-height: 600px;
                    max-height: 800px;
                    border-radius: 10px;
                    margin: 10px;
                }}
                QWidget#contentOverlay {{
                    border-radius: 10px;
                    margin: 20px;
                    background-color: {p["window_bg"]}cc;  /* Semi-transparent background */
                }}
                QGroupBox#statsGroup {{
                    background-color: {p["group_bg"]}a0;
                    border: 1px solid {p["border_color"]};
                    border-radius: 8px;
                }}
            """
            self.log(f"Background image loaded from: {background_image_url}")
        else:
            # Fallback style if image not found
            background_style = f"""
                 /* Ensure containers are transparent even without image if needed */
                 QWidget#backgroundContainer, QWidget#HomePage, QWidget#contentOverlay, QGroupBox#statsGroup {{ 
                     background-color: transparent; 
                     border: none; 
                 }}
                 QWidget#backgroundContainer {{
                     min-height: 600px;
                     max-height: 800px;
                     border-radius: 10px;
                     margin: 10px;
                 }}
            """
            self.log(f"Background image not found at: {bg_path}")

        qss = f"""
            {background_style}
            QMainWindow, QWidget#central_widget {{ background-color: {p["window_bg"]}; }}
            QWidget#HomePage, QWidget#DataPage, QWidget#TrainPage, QWidget#DeployPage, QWidget#SettingsPage, QWidget#AboutPage {{
                background-color: {p["window_bg"]}; 
            }}
            QScrollArea#pageScrollArea {{ background-color: transparent; border: none; }}
            QScrollArea#pageScrollArea > QWidget > QWidget {{ background-color: transparent; }} 
            
            QWidget#backgroundContainer {{
                background-color: transparent;
            }}

            /* About Page Specific Styling */
            QWidget#aboutContentContainer {{
                background-color: {p["group_bg"]};
                border-radius: 10px;
                border: 1px solid {p["border_color"]};
            }}
            QLabel#pageTitle {{ 
                font-size: 24pt; 
                font-weight: bold; 
                color: {p["label_header_color"]}; 
                padding: 10px;
                margin: 10px 0;
            }}
            QLabel#versionLabel {{
                font-size: 11pt;
                color: {p["text_color"]};
                opacity: 0.8;
            }}
            QLabel#introText {{ 
                font-size: 11pt; 
                line-height: 1.5; 
                color: {p["text_color"]}; 
            }}

            QWidget#nav_widget {{ 
                background-color: {p["nav_bar_bg"]}; 
                border-top: 1px solid {p["border_color"]}; 
            }}
            QPushButton.navButton {{
                background-color: {p["nav_button_bg"]}; color: {p["nav_button_text"]};
                border: none; padding: 10px; font-size: 10pt; font-weight: bold;
            }}
            QPushButton.navButton:hover {{ background-color: {p["nav_button_hover_bg"]}; }}
            QPushButton.navButton:checked {{
                background-color: {p["nav_button_checked_bg"]};
                border-bottom: 3px solid {p["label_header_color"]}; 
            }}

            QGroupBox {{
                background-color: {p["group_bg"]}; border: 1px solid {p["border_color"]};
                border-radius: 8px; margin-top: 12px; padding: 10px; 
                font-weight: bold; color: {p["text_color"]};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0px 8px 5px 8px; 
                color: {p["label_header_color"]}; font-size: 11pt;
                font-weight: bold; 
            }}

            QLabel, QCheckBox {{ color: {p["text_color"]}; font-size: 9pt; }}
            QLabel#infoLabel {{ font-size: 10pt; font-weight: bold; color: {p["label_header_color"]}; }}
            
            /* Base button styling for all buttons */
            QPushButton {{
                background-color: {p["button_bg"]};
                color: {p["button_text"]};
                border: 1px solid {p["border_color"]};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 10pt;
                font-weight: 500;
                min-width: 120px;
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
                border-color: {p["disabled_text"]};
            }}

            /* Special button styles */
            QPushButton.navButton {{
                background-color: {p["nav_button_bg"]};
                color: {p["nav_button_text"]};
                border: none;
                padding: 10px;
                font-size: 10pt;
                font-weight: bold;
                min-width: 80px;
            }}
            
            QPushButton.navButton:hover {{
                background-color: {p["nav_button_hover_bg"]};
            }}
            
            QPushButton.navButton:checked {{
                background-color: {p["nav_button_checked_bg"]};
                border-bottom: 3px solid {p["label_header_color"]};
            }}

            QPushButton#consoleToggle {{
                background-color: {p["button_bg"]};
                border: none;
                border-radius: 10px;
                padding: 0px;
                margin: 0px;
                min-width: 36px;
                max-width: 36px;
            }}

            QPushButton#consoleToggle::before {{
                content: '';
                position: absolute;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                background-color: white;
                top: 2px;
                left: 2px;
                transition: left 0.2s ease-in-out;
            }}

            QPushButton#consoleToggle:checked {{
                background-color: {p["label_header_color"]};
            }}

            QPushButton#consoleToggle:checked::before {{
                left: 18px;
            }}

            QPushButton#consoleToggle:hover {{
                background-color: {p["button_hover_bg"]};
            }}

            QPushButton#consoleToggle:checked:hover {{
                background-color: {p["button_hover_bg"]};
            }}

            QComboBox, QSpinBox, QLineEdit {{
                background-color: {p["input_bg"]}; color: {p["text_color"]};
                border: 1px solid {p["input_border"]}; padding: 6px; 
                border-radius: 4px; min-height: 22px; 
            }}
            QComboBox::drop-down {{ border: none; subcontrol-origin: padding; subcontrol-position: top right; width: 15px; }}
            QComboBox::down-arrow {{ image: url(icons/{self.current_theme}/dropdown_arrow.png); }} 
            QComboBox QAbstractItemView {{ 
                background-color: {p["input_bg"]}; color: {p["text_color"]};
                border: 1px solid {p["input_border"]}; 
                selection-background-color: {p["label_header_color"]}; 
                selection-color: {p["window_bg"] if self.current_theme == 'dark' else p["text_color"]}; 
            }}

            QProgressBar {{
                border: 1px solid {p["border_color"]}; border-radius: 4px;
                text-align: center; color: {p["text_color"]};
                background-color: {p["input_bg"]}; 
            }}
            QProgressBar::chunk {{ 
                background-color: {p["label_header_color"]}; 
                border-radius: 3px; 
                margin: 1px; 
            }}

            QTextEdit {{ 
                background-color: {p["log_bg"]}; color: {p["log_text"]};
                border: 1px solid {p["border_color"]}; border-radius: 4px;
                font-family: "Monaco", "Consolas", "Courier New", monospace; font-size: 9pt;
            }}
            QWidget#logAreaWidget {{ background-color: {p["window_bg"]}; }} 

            QScrollBar:vertical {{ 
                border: none; background: {p["scroll_bar_bg"]}; 
                width: 12px; margin: 0px; border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{ 
                background: {p["scroll_bar_handle"]}; min-height: 25px; 
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ 
                border: none; background: none; height: 0px; 
            }}
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{ background: none; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}


            QScrollBar:horizontal {{ 
                border: none; background: {p["scroll_bar_bg"]}; 
                height: 12px; margin: 0px; border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{ 
                background: {p["scroll_bar_handle"]}; min-width: 25px; 
                border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ 
                 border: none; background: none; width: 0px; 
            }}
            QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{ background: none; }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

            QWidget#videoContainer {{
                background-color: {p["group_bg"]};
                border: none;
                margin: 0px;
                padding: 0px;
            }}

            QWidget#toggleContainer {{
                background-color: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }}

            /* Download Button Styling */
            QPushButton#downloadButton {{
                background-color: {p["button_bg"]};
                color: {p["button_text"]};
                border: 1px solid {p["border_color"]};
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 10pt;
                font-weight: 500;
                min-width: 120px;
            }}
            
            QPushButton#downloadButton:hover {{
                background-color: {p["button_hover_bg"]};
            }}
            
            QPushButton#downloadButton:pressed {{
                background-color: {p["button_pressed_bg"]};
            }}
            
            QPushButton#downloadButton:disabled {{
                background-color: {p["disabled_bg"]};
                color: {p["disabled_text"]};
                border-color: {p["disabled_text"]};
            }}

            QWidget#downloadContainer {{
                background-color: transparent;
                border: none;
            }}

            QLabel#downloadSpinner {{
                background-color: transparent;
                border: none;
            }}

            QProgressBar#downloadProgressBar {{
                border: 1px solid {p["border_color"]};
                border-radius: 4px;
                text-align: center;
                color: {p["text_color"]};
                background-color: {p["input_bg"]};
                min-height: 20px;
                max-height: 20px;
            }}
            
            QProgressBar#downloadProgressBar::chunk {{
                background-color: {p["label_header_color"]};
                border-radius: 3px;
                margin: 1px;
            }}
        """
        self.setStyleSheet(qss)
        self.update_all_icons()

    def change_theme(self, theme_name_full):
        new_theme_base = theme_name_full.lower().split(" ")[0]
        if self.current_theme != new_theme_base:
            self.log(f"Changing theme to {theme_name_full}")
            # Update current_theme before applying stylesheet so get_icon uses the new theme
            self.current_theme = new_theme_base
            self.apply_stylesheet(
                self.current_theme
            )  # apply_stylesheet will now use the updated self.current_theme

            # Rebuild About page to reflect new theme colors in HTML links
            # This is a bit heavy, but ensures dynamic HTML content is updated.
            # A more granular update would require storing references to the specific labels.
            if hasattr(self, "stacked_widget") and hasattr(self, "about_page_index"):
                old_about_page = self.stacked_widget.widget(self.about_page_index)
                if old_about_page:
                    self.stacked_widget.removeWidget(old_about_page)
                    old_about_page.deleteLater()
                self.about_page = self.create_about_page()
                self.stacked_widget.insertWidget(self.about_page_index, self.about_page)
                # Ensure current page is maintained if it was the about page
                # self.stacked_widget.setCurrentIndex(self.about_page_index) # Or restore previous index

    def log(self, message):
        current_time_str = time.strftime("%H:%M:%S")
        log_message_formatted = f"[{current_time_str}] {message}"

        if hasattr(self, "log_text") and self.log_text:
            if self.log_group.isVisible():
                self.log_text.append(log_message_formatted)
                self.log_text.verticalScrollBar().setValue(
                    self.log_text.verticalScrollBar().maximum()
                )
            else:
                print(f"LOG_HIDDEN: {log_message_formatted}")
        else:
            print(f"LOG_EARLY: {log_message_formatted}")

    def update_ultralytics_status_display(self):
        if ULTRALYTICS_INSTALLED:
            self.ultralytics_status_label.setText("Ultralytics Status: Installed ✔️")
            self.install_ultralytics_button.setText(" Verify/Reinstall Ultralytics")
            self.install_ultralytics_button.setEnabled(True)
        else:
            self.ultralytics_status_label.setText(
                "Ultralytics Status: Not Installed ❌"
            )
            self.install_ultralytics_button.setText(" Install Ultralytics")
            self.install_ultralytics_button.setEnabled(True)

        if hasattr(self, "train_button") and self.train_button:
            self.train_button.setEnabled(
                ULTRALYTICS_INSTALLED and self.data_yaml_path is not None
            )

        if hasattr(self, "update_inference_controls_state"):
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
        if device_text == "cpu":
            return "cpu"
        elif "cuda" in device_text:
            return device_text.split(" ")[0]
        return "cpu"

    def update_webcam_list(self):
        current_selection = self.webcam_combo.currentText()
        self.webcam_combo.clear()
        cams = get_available_cameras()
        if cams:
            self.webcam_combo.addItems([f"Webcam {i}" for i in cams])
            # Try to restore previous selection if still available
            if current_selection in [f"Webcam {i}" for i in cams]:
                self.webcam_combo.setCurrentText(current_selection)
            elif self.webcam_combo.count() > 0:
                self.webcam_combo.setCurrentIndex(
                    0
                )  # Default to first if previous not found

            if hasattr(self, "webcam_resolution_combo"):
                self.webcam_resolution_combo.setEnabled(True)
            # start_webcam_button enablement depends on model loaded state too, handled in update_inference_controls_state
        else:
            self.webcam_combo.addItem("No webcams found")
            self.webcam_combo.setEnabled(False)
            if hasattr(self, "webcam_resolution_combo"):
                self.webcam_resolution_combo.setEnabled(False)
        self.update_inference_controls_state()  # Refresh dependent controls

    def update_start_webcam_button_state(self):
        if hasattr(self, "start_webcam_button"):  # Ensure button exists
            if (
                self.inference_thread
                and self.inference_thread.isRunning()
                and self.inference_thread.source_type == "webcam"
            ):
                self.start_webcam_button.setText(" Stop Webcam")
                self.start_webcam_button.setIcon(self.get_icon("webcam_stop"))
                self.start_webcam_button.setToolTip("Stop live webcam inference")
            else:
                self.start_webcam_button.setText(" Start Webcam")
                self.start_webcam_button.setIcon(self.get_icon("webcam_start"))
                self.start_webcam_button.setToolTip("Start live webcam inference")

    def update_inference_controls_state(self):
        model_loaded = bool(self.trained_model_path_for_inference)
        ultralytics_ready = ULTRALYTICS_INSTALLED

        can_perform_actions_with_model = model_loaded and ultralytics_ready
        webcam_available = (
            self.webcam_combo.count() > 0
            and "No webcams found" not in self.webcam_combo.currentText()
        )

        is_any_inference_running = (
            self.inference_thread is not None and self.inference_thread.isRunning()
        )
        is_webcam_inference_running = (
            is_any_inference_running and self.inference_thread.source_type == "webcam"
        )
        is_file_inference_running = (
            is_any_inference_running
            and self.inference_thread.source_type in ["video", "image"]
        )

        if hasattr(self, "start_webcam_button"):
            self.start_webcam_button.setEnabled(
                can_perform_actions_with_model
                and webcam_available
                and not is_file_inference_running
            )
            self.update_start_webcam_button_state()

        if hasattr(self, "webcam_resolution_combo"):
            self.webcam_resolution_combo.setEnabled(
                can_perform_actions_with_model
                and webcam_available
                and not is_any_inference_running
            )
        if hasattr(self, "load_media_button"):
            self.load_media_button.setEnabled(
                can_perform_actions_with_model and not is_any_inference_running
            )
        if hasattr(self, "stop_inference_button"):
            self.stop_inference_button.setEnabled(is_file_inference_running)

        if hasattr(self, "export_model_button"):
            self.export_model_button.setEnabled(
                can_perform_actions_with_model and not is_any_inference_running
            )
        if hasattr(self, "load_trained_model_button"):
            self.load_trained_model_button.setEnabled(not is_any_inference_running)

        if not model_loaded:
            if hasattr(self, "inference_model_status_label"):
                self.inference_model_status_label.setText(
                    "No model loaded. Please load a model file."
                )
            if hasattr(self, "export_status_label"):
                self.export_status_label.setText("Export Status: Load a model first.")
        elif not ultralytics_ready:
            if hasattr(self, "inference_model_status_label"):
                self.inference_model_status_label.setText(
                    "Ultralytics not installed. Please install from Settings."
                )
            if hasattr(self, "export_status_label"):
                self.export_status_label.setText(
                    "Export Status: Ultralytics not installed."
                )
        else:
            if not is_any_inference_running:
                if hasattr(self, "inference_model_status_label"):
                    self.inference_model_status_label.setText(
                        f"Active Model: {os.path.basename(self.trained_model_path_for_inference if self.trained_model_path_for_inference else 'N/A')}"
                    )
                if hasattr(self, "export_status_label"):
                    self.export_status_label.setText("Export Status: Ready")

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
        self.populate_device_combo()

    def install_specific_dependency(self, dep_name, model_path_context):
        self.log(f"Starting installation process for: {dep_name}")
        # Disable inference buttons during install
        if hasattr(self, "start_webcam_button"):
            self.start_webcam_button.setEnabled(False)
        if hasattr(self, "load_media_button"):
            self.load_media_button.setEnabled(False)
        # Potentially disable settings page buttons too if complex

        self.dependency_install_thread = InstallSpecificDependencyThread(
            dep_name, model_path_context
        )
        self.dependency_install_thread.log_message.connect(self.log)
        self.dependency_install_thread.finished.connect(
            self.on_specific_dependency_install_finished
        )
        self.dependency_install_thread.start()

    def on_specific_dependency_install_finished(
        self, success, dep_name, model_path_context
    ):
        self.log(f"Installation of {dep_name} {'succeeded' if success else 'failed'}.")
        self.update_inference_controls_state()  # Re-evaluate and enable buttons

        if success:
            QMessageBox.information(
                self,
                "Installation Successful",
                f"{dep_name} has been installed. Please try loading the model "
                f"'{os.path.basename(model_path_context)}' again for inference.",
            )
        else:
            QMessageBox.warning(
                self,
                "Installation Failed",
                f"Failed to install {dep_name}. Please check the console log. "
                "You may need to install it manually (e.g., using pip) and then restart the application or try loading the model again.",
            )
        self.dependency_install_thread = None

    def select_dataset_zip(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset Zip File", "", "Zip Files (*.zip)", options=options
        )
        if fileName:
            self.dataset_zip_path = fileName
            self.data_zip_label.setText(f"Selected: {os.path.basename(fileName)}")
            self.log(f"Dataset ZIP selected: {fileName}")
            self.data_yaml_path = None
            self.data_yaml_path_label.setText("Data YAML: Not generated yet.")
            if hasattr(self, "train_button"):
                self.train_button.setEnabled(False)
            self.start_data_preparation()
        else:
            self.log("Dataset selection cancelled.")

    def start_data_preparation(self):
        if not self.dataset_zip_path:
            self.log("Error: No dataset ZIP file selected.")
            self.data_prep_status_label.setText("Status: Select a dataset first ❌")
            return

        self.log("Starting data preparation thread...")
        self.data_prep_status_label.setText("Status: Preparing dataset...")
        self.data_prep_progress.setValue(0)

        data_processing_subdir = os.path.join(
            self.work_dir,
            "dataset_processing",
            os.path.splitext(os.path.basename(self.dataset_zip_path))[0],
        )
        try:
            if os.path.exists(data_processing_subdir):
                shutil.rmtree(data_processing_subdir)  # Clean before use
            os.makedirs(data_processing_subdir, exist_ok=True)
        except OSError as e:
            self.log(f"Error managing data processing directory: {e}")
            self.data_prep_status_label.setText(f"Status: Directory error ❌")
            return

        self.data_prep_thread = DataPrepThread(
            self.dataset_zip_path, data_processing_subdir
        )
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
            self.data_yaml_path_label.setText(
                f"Data YAML: {os.path.basename(self.data_yaml_path)}"
            )
            if hasattr(self, "train_button"):
                self.train_button.setEnabled(ULTRALYTICS_INSTALLED)
        else:
            self.log(f"Data preparation failed: {path_or_message}")
            self.data_prep_status_label.setText(
                f"Status: Failed ❌ ({path_or_message})"
            )
            self.data_yaml_path_label.setText("Data YAML: Generation failed.")
            if hasattr(self, "train_button"):
                self.train_button.setEnabled(False)
        self.data_prep_progress.setValue(100 if success else 0)

    def start_training_process(self):
        if not self.data_yaml_path:
            self.log("Error: Dataset not prepared or YAML path not set.")
            self.train_status_label.setText("Status: Prepare dataset first ❌")
            QMessageBox.warning(
                self,
                "Dataset Missing",
                "Please prepare a dataset on the 'Data' tab first.",
            )
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics is not installed. Cannot start training.")
            self.train_status_label.setText(
                "Status: Install Ultralytics from Settings ❌"
            )
            QMessageBox.warning(
                self,
                "Ultralytics Missing",
                "Ultralytics is not installed. Please install it from the 'Settings' tab.",
            )
            return

        if hasattr(self, "train_button"):
            self.train_button.setEnabled(False)
        self.train_status_label.setText("Status: Starting training...")
        self.log("Initiating training...")

        model_to_train = self.train_model_combo.currentText()
        epochs = self.epochs_spinbox.value()
        imgsz = self.imgsz_spinbox.value()
        device_to_use = self.get_selected_device()
        project_name = self.train_project_name_input.currentText().strip()
        if not project_name:
            project_name = "yolo_gui_runs"
            self.log("Project name was empty, defaulting to 'yolo_gui_runs'.")
            self.train_project_name_input.setCurrentText(project_name)

        self.train_thread = TrainThread(
            model_to_train,
            self.data_yaml_path,
            epochs,
            imgsz,
            device_to_use,
            project_name,
        )
        self.train_thread.progress_update.connect(self.log)
        self.train_thread.finished.connect(self.on_training_session_finished)
        self.train_thread.start()

    def on_training_session_finished(self, success, model_path_or_message):
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(
                ULTRALYTICS_INSTALLED and self.data_yaml_path is not None
            )
        if success:
            self.log(f"Training successful! Model saved at: {model_path_or_message}")
            self.train_status_label.setText(f"Status: Training Complete ✔️")
            self.trained_model_path_display_label.setText(
                f"Latest Trained Model: {os.path.basename(model_path_or_message)}"
            )

            if self.load_model_after_train_checkbox.isChecked():
                self.log(
                    f"Automatically loading trained model '{os.path.basename(model_path_or_message)}' into Deploy tab."
                )
                self.select_model_for_inference(model_path=model_path_or_message)
                # Find deploy page index dynamically
                deploy_page_idx = -1
                for item in self.nav_buttons_data:
                    if item["text"] == "Deploy":
                        deploy_page_idx = item["page_index"]
                        break
                if deploy_page_idx != -1:
                    self.switch_page(deploy_page_idx)
                else:
                    self.log("Could not find Deploy page index to switch.")

        else:
            self.log(f"Training failed: {model_path_or_message}")
            self.train_status_label.setText(f"Status: Training Failed ❌")
            self.trained_model_path_display_label.setText(
                "Trained Model: Error during training."
            )

    def select_model_for_inference(self, model_path=None):
        if not isinstance(model_path, str) or not model_path:
            options = QFileDialog.Options()
            file_filter = get_inference_model_extensions_filter()
            fileName, _ = QFileDialog.getOpenFileName(
                self, "Select Model File", "", file_filter, options=options
            )
            if not fileName:
                self.log("Model selection cancelled.")
                return
            model_path = fileName

        self.trained_model_path_for_inference = model_path
        self.log(
            f"Model selected for inference/export: {self.trained_model_path_for_inference}"
        )
        self.inference_model_status_label.setText(
            f"Active Model: {os.path.basename(self.trained_model_path_for_inference)}"
        )
        self.update_inference_controls_state()
        self.video_feed_label.setText("Output Preview (Model Loaded)")
        self.fps_label.setText("FPS: --")
        self.detected_objects_label.setText("Objects: --")

    def toggle_webcam_inference_action(self):
        if (
            self.inference_thread
            and self.inference_thread.isRunning()
            and self.inference_thread.source_type == "webcam"
        ):
            self.stop_live_inference()
        else:
            if self.inference_thread and self.inference_thread.isRunning():
                QMessageBox.warning(
                    self,
                    "Inference Busy",
                    "Another inference (e.g., from a file) is currently running. Please stop it first.",
                )
                return
            self.run_webcam_inference()
        # update_inference_controls_state is called by run_webcam_inference or on_live_inference_finished

    def run_live_inference(self, source_type, source_path):
        if not self.trained_model_path_for_inference:
            self.log("Error: No trained model loaded for inference.")
            self.video_feed_label.setText("Error: No model loaded!")
            QMessageBox.warning(
                self,
                "Model Missing",
                "No model is loaded. Please load a model on the 'Deploy' tab.",
            )
            return
        if not ULTRALYTICS_INSTALLED:
            self.log(
                "Error: Ultralytics is not installed. Required for YOLO interface."
            )
            self.video_feed_label.setText("Error: Ultralytics not installed!")
            QMessageBox.warning(
                self,
                "Ultralytics Missing",
                "Ultralytics library is not installed. Please install it from the Settings page to enable inference.",
            )
            return
        if self.inference_thread and self.inference_thread.isRunning():
            self.log("Inference is already running. Stop it first.")
            QMessageBox.information(
                self,
                "Busy",
                "An inference process is already running. Please stop it first.",
            )
            return
        if (
            self.dependency_install_thread
            and self.dependency_install_thread.isRunning()
        ):
            self.log("Dependency installation in progress. Please wait.")
            QMessageBox.information(
                self,
                "Busy",
                "A dependency installation is currently in progress. Please wait until it finishes.",
            )
            return

        self.log(f"Starting live inference on {source_type}: {source_path}")
        self.video_feed_label.setText("Starting inference...")
        device_to_use = self.get_selected_device()

        selected_resolution_text = self.webcam_resolution_combo.currentText()
        webcam_res = (
            self.webcam_resolutions.get(selected_resolution_text)
            if source_type == "webcam"
            else None
        )

        self.inference_thread = InferenceThread(
            self.trained_model_path_for_inference,
            source_type,
            source_path,
            device_to_use,
            resolution=webcam_res,
        )
        self.inference_thread.log_message.connect(self.log)
        self.inference_thread.frame_ready.connect(self.update_inference_video_feed)
        self.inference_thread.finished.connect(self.on_live_inference_finished)

        self.inference_thread.start()
        self.update_inference_controls_state()

    def run_webcam_inference(self):
        cam_text = self.webcam_combo.currentText()
        if "No webcams" in cam_text or not cam_text:
            self.log("No webcam selected or available.")
            QMessageBox.warning(
                self, "Webcam Error", "No webcam selected or available."
            )
            return
        try:
            cam_index_str = cam_text.split()[-1]
            int(cam_index_str)
            self.run_live_inference("webcam", cam_index_str)
        except (IndexError, ValueError):
            self.log(f"Invalid webcam selection: {cam_text}")
            QMessageBox.warning(
                self, "Webcam Error", f"Invalid webcam selection: {cam_text}"
            )

    def run_file_inference(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video or Image File",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.webp);;Video Files (*.mp4 *.avi *.mov *.mkv);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp)",
            options=options,
        )
        if fileName:
            img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            if fileName.lower().endswith(img_extensions):
                self.run_live_inference("image", fileName)
            else:
                self.run_live_inference("video", fileName)

    def stop_live_inference(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self.log(
                f"Sending stop signal to {self.inference_thread.source_type} inference process..."
            )
            self.inference_thread.stop()
        else:
            self.log("No active inference process to stop.")
        # UI update will be handled by on_live_inference_finished

    def update_inference_video_feed(self, q_image, fps, object_count):
        if not self.video_feed_label.pixmap():
            self.video_feed_label.setPixmap(QPixmap(self.video_feed_label.size()))

        # Scale QImage to fit the fixed size of the label while maintaining aspect ratio
        scaled_image = q_image.scaled(
            640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_feed_label.setPixmap(QPixmap.fromImage(scaled_image))
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.detected_objects_label.setText(f"Objects: {object_count}")

    def on_live_inference_finished(self, message, source_type):
        self.log(f"Inference process feedback ({source_type}): {message}")

        if message.startswith("missing_dependency:"):
            parts = message.split(":", 2)
            if len(parts) == 3:
                _, dep_name, model_path_context = parts
                self.video_feed_label.setText(f"Output Preview ({dep_name} Missing)")
                self.fps_label.setText("FPS: --")
                self.detected_objects_label.setText("Objects: --")

                reply = QMessageBox.question(
                    self,
                    "Missing Dependency",
                    f"The backend dependency '{dep_name}' is required for the model "
                    f"'{os.path.basename(model_path_context)}' but it appears to be missing or not working.\n\n"
                    f"Would you like to attempt an automatic installation of '{dep_name}' via pip?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self.install_specific_dependency(dep_name, model_path_context)
                else:
                    self.log(
                        f"Installation of {dep_name} skipped by user. Please install it manually."
                    )
                    QMessageBox.information(
                        self,
                        "Manual Installation Required",
                        f"Please install '{dep_name}' manually (e.g., using 'pip install {dep_name.split('-')[0]}') "  # Suggest base package name
                        "and try loading the model again.",
                    )
            else:
                self.video_feed_label.setText(
                    f"Output Preview (Error parsing dependency message)"
                )
            self.inference_thread = None
            self.update_inference_controls_state()
            return

        if "error" in message.lower():
            self.video_feed_label.setText(f"Output Preview (Error Occurred)\n{message}")
        elif message == "Image inference complete.":
            self.log("Image processing displayed on video feed.")
            # The image is already displayed by update_inference_video_feed.
            # No need to change video_feed_label text here to keep the image visible.
        elif message == "Inference stopped.":
            self.video_feed_label.setText(f"Output Preview (Stopped)")
        else:
            self.video_feed_label.setText(f"Output Preview\n({message})")

        if (
            message != "Image inference complete."
        ):  # Don't clear stats for a successfully displayed image
            self.fps_label.setText("FPS: --")
            self.detected_objects_label.setText("Objects: --")

        self.inference_thread = None
        self.update_inference_controls_state()

    def run_export_model(self):
        if not self.trained_model_path_for_inference:
            self.log("Error: No trained model loaded to export.")
            self.export_status_label.setText("Export Status: Load a model first ❌")
            QMessageBox.warning(
                self,
                "Model Missing",
                "No model loaded. Please load a model before exporting.",
            )
            return
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics is not installed for export.")
            self.export_status_label.setText(
                "Export Status: Ultralytics not installed ❌"
            )
            QMessageBox.warning(
                self,
                "Ultralytics Missing",
                "Ultralytics is not installed. Please install it from Settings.",
            )
            return

        export_format_selected = self.export_format_combo.currentText()
        self.log(
            f"Preparing to export model {os.path.basename(self.trained_model_path_for_inference)} to {export_format_selected} format..."
        )
        self.export_status_label.setText(
            f"Status: Exporting to {export_format_selected}..."
        )
        if hasattr(self, "export_model_button"):
            self.export_model_button.setEnabled(False)

        self.export_thread = ExportThread(
            self.trained_model_path_for_inference, export_format_selected
        )
        self.export_thread.log_message.connect(self.log)
        self.export_thread.finished.connect(self.on_export_model_finished)
        self.export_thread.start()

    def on_export_model_finished(self, success, path_or_message):
        if success:
            self.log(f"Export successful: {path_or_message}")
            self.export_status_label.setText(
                f"Status: Export Successful ✔️ ({os.path.basename(path_or_message)})"
            )
        else:
            self.log(f"Export failed: {path_or_message}")
            self.export_status_label.setText(f"Status: Export Failed ❌")
        if hasattr(self, "export_model_button"):
            self.export_model_button.setEnabled(True)

    def closeEvent(self, event):
        self.log("Application closing. Stopping active threads...")
        threads_to_stop = [
            "inference_thread",
            "dependency_install_thread",
            "train_thread",
            "data_prep_thread",
            "install_thread",
            "export_thread",
        ]
        for thread_attr_name in threads_to_stop:
            thread_instance = getattr(self, thread_attr_name, None)
            if thread_instance and thread_instance.isRunning():
                self.log(f"Stopping {thread_attr_name}...")
                if hasattr(thread_instance, "stop") and callable(
                    getattr(thread_instance, "stop")
                ):
                    thread_instance.stop()  # Custom stop method
                else:
                    thread_instance.quit()  # Standard QThread quit

                if not thread_instance.wait(2000):  # Wait up to 2 seconds
                    self.log(
                        f"{thread_attr_name} did not stop gracefully. Terminating..."
                    )
                    thread_instance.terminate()
                    thread_instance.wait()  # Wait for termination
                else:
                    self.log(f"{thread_attr_name} stopped.")
            setattr(self, thread_attr_name, None)  # Clear attribute

        self.log("Exiting VisionCraft Studio.")
        event.accept()

    def download_selected_model(self):
        if not ULTRALYTICS_INSTALLED:
            self.log("Error: Ultralytics not installed. Please install from Settings.")
            return

        model_name = self.train_model_combo.currentText()
        models_dir = os.path.join("assets", "models")
        os.makedirs(models_dir, exist_ok=True)
        target_path = os.path.join(models_dir, model_name)

        if os.path.exists(target_path):
            self.log(f"Model {model_name} already exists in {models_dir}. Downloading again...")

        # Show loading state
        self.download_model_button.setEnabled(False)
        self.download_model_button.setText(" Downloading...")
        self.download_spinner.show()
        self.download_spinner.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-top: 2px solid #007AFF;
                border-radius: 50%;
            }
        """)

        # Create progress bar if it doesn't exist
        if not hasattr(self, 'download_progress_bar'):
            self.download_progress_bar = QProgressBar()
            self.download_progress_bar.setObjectName("downloadProgressBar")
            self.download_progress_bar.setRange(0, 100)
            self.download_progress_bar.setTextVisible(True)
            self.download_progress_bar.setFormat("%p% (%v/%m)")
            # Add it to the layout after the download button
            self.train_model_combo.parent().layout().addWidget(self.download_progress_bar)

        self.download_progress_bar.setValue(0)
        self.download_progress_bar.show()

        # Create and start download thread
        self.download_thread = ModelDownloadThread(model_name, target_path)
        self.download_thread.log_message.connect(self.log)
        self.download_thread.progress.connect(self.update_download_progress)
        self.download_thread.finished.connect(self.on_model_download_finished)
        self.download_thread.start()

    def update_download_progress(self, message, value):
        if hasattr(self, 'download_progress_bar'):
            self.download_progress_bar.setValue(value)
            self.download_progress_bar.setFormat(f"{message} %p%")

    def on_model_download_finished(self, success, path_or_error):
        # Reset button state
        self.download_model_button.setEnabled(True)
        self.download_model_button.setText(" Download Selected Model")
        self.download_spinner.hide()
        
        if hasattr(self, 'download_progress_bar'):
            self.download_progress_bar.hide()
        
        if not success:
            self.log(f"✗ Download failed: {path_or_error}")
        else:
            self.log(f"✓ Model downloaded successfully to {path_or_error}")


# --- Icon Creation (Placeholder - User should provide actual icons) ---
def create_dummy_icons_if_needed():
    icon_base_dir = "icons"
    themes = ["light", "dark"]
    dummy_icon_list = [
        "home",
        "data",
        "train",
        "deploy",
        "settings",
        "about",
        "app_icon",
        "console_show",
        "console_hide",
        "dropdown_arrow",
        "dataset_load",
        "train_action",
        "model_load",
        "media_load",
        "webcam_start",
        "webcam_stop",
        "stop_inference",
        "model_export",
        "yolo_icon",
        "model_download",  # Added new icon
    ]

    for theme in themes:
        theme_dir = os.path.join(icon_base_dir, theme)
        if not os.path.exists(theme_dir):
            try:
                os.makedirs(theme_dir)
                print(f"Created dummy icon directory: {theme_dir}")
            except OSError as e:
                print(f"Error creating directory {theme_dir}: {e}")
                continue  # Skip this theme if dir creation fails

        for icon_name_base in dummy_icon_list:
            icon_file = os.path.join(theme_dir, icon_name_base + ".png")
            if not os.path.exists(icon_file):
                try:
                    img = QImage(24, 24, QImage.Format_ARGB32)
                    img.fill(
                        Qt.transparent
                        if icon_name_base != "app_icon"
                        else (Qt.blue if theme == "light" else Qt.darkGray)
                    )

                    # Simple visual cue for dummy icons
                    # painter = QPainter(img)
                    # painter.setPen(Qt.black if theme == "light" else Qt.white)
                    # painter.drawText(img.rect(), Qt.AlignCenter, icon_name_base[:1]) # Draw first letter
                    # painter.end()

                    if img.save(icon_file):
                        pass  # print(f"Created dummy icon: {icon_file}")
                    else:
                        print(f"Failed to save dummy icon: {icon_file}")
                except Exception as e:
                    print(f"Could not create dummy icon {icon_file}: {e}")

    generic_app_icon = os.path.join(icon_base_dir, "app_icon.png")
    if not os.path.exists(generic_app_icon):
        try:
            img = QImage(32, 32, QImage.Format_ARGB32)
            img.fill(Qt.darkCyan)
            if not img.save(generic_app_icon):
                print(f"Failed to save dummy generic app_icon.png")
        except Exception as e:
            print(f"Error creating generic app_icon: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create dummy icons folder and files if they don't exist for testing UI
    dummy_icon_dir = "icons"
    if not os.path.exists(dummy_icon_dir):
        os.makedirs(dummy_icon_dir)
        dummy_icons = [
            "home.png",
            "data.png",
            "train.png",
            "deploy.png",
            "settings.png",
            "about.png",
            "console_show.png",
            "console_hide.png",
            "dropdown_arrow.png",
            "dataset_load.png",
            "train_action.png",
            "model_load.png",
            "media_load.png",
            "webcam_start.png",
            "webcam_stop.png",
            "stop_inference.png",
            "model_export.png",
            "yolo_icon.png",
        ]
        for icon_name in dummy_icons:
            try:
                # Create tiny placeholder png using QImage
                img = QImage(16, 16, QImage.Format_RGB32)
                img.fill(Qt.gray)
                img.save(os.path.join(dummy_icon_dir, icon_name))
            except Exception as e:
                print(f"Could not create dummy icon {icon_name}: {e}")

    window = ModernYoloGUI()
    window.show()
    sys.exit(app.exec_())
