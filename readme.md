# VisionCraft Studio ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Deep%20Learning-YOLOv8-purple.svg)](https://ultralytics.com/)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/your-username/vision-craft-studio) <!-- Replace with your actual version and link -->
[![Issues](https://img.shields.io/github/issues/your-username/vision-craft-studio.svg)](https://github.com/your-username/vision-craft-studio/issues) <!-- Replace with your username/repo -->
[![Forks](https://img.shields.io/github/forks/your-username/vision-craft-studio.svg)](https://github.com/your-username/vision-craft-studio/network/members) <!-- Replace with your username/repo -->
[![Stars](https://img.shields.io/github/stars/your-username/vision-craft-studio.svg)](https://github.com/your-username/vision-craft-studio/stargazers) <!-- Replace with your username/repo -->

<!-- Optional: Add a logo here -->
<!-- <p align="center">
  <img src="link_to_your_logo.png" alt="VisionCraft Studio Logo" width="200"/>
</p> -->

VisionCraft Studio is a user-friendly desktop application designed to simplify the process of creating, training, and deploying custom deep learning YOLO (You Only Look Once) object detection and segmentation models. With an intuitive Graphical User Interface (GUI), users can manage datasets, train models, perform real-time inference, and export models to various formats without writing extensive code.

## 🌟 Features

* **Intuitive GUI:** Easy-to-navigate interface built with PyQt5.
* **Dataset Management:**
    * Load custom datasets from `.zip` files (images and YOLO format labels).
    * Automatic splitting into training and validation sets.
    * Generation of `data.yaml` for YOLO training.
* **Model Training:**
    * Train YOLOv8 detection and segmentation models.
    * Select from various pre-trained YOLOv8 backbones (e.g., yolov8n, yolov8s, yolov8m).
    * Customize training parameters: epochs, image size, compute device (CPU/GPU).
    * Organized project runs for training outputs.
* **Model Deployment & Inference:**
    * Load custom trained `.pt` models or other compatible formats.
    * Real-time inference via webcam.
    * Inference on local video or image files.
    * Adjustable webcam resolution and confidence thresholds.
    * Display FPS and detected object counts.
* **Model Export:**
    * Convert trained PyTorch models to various formats including ONNX, TorchScript, CoreML, TensorFlow Lite, OpenVINO, and more.
* **Dependency Management:**
    * In-app installation/verification for the Ultralytics library.
    * Assisted installation for model-specific backends (e.g., ONNX Runtime, OpenVINO).
* **Customization:**
    * Switch between Light and Dark (Dracula) UI themes.
    * Configurable default compute device.
* **User-Friendly Output:**
    * Integrated console log for monitoring processes.
    * Clear status updates and progress bars.

## 🎬 Demo

<!-- Replace with a GIF or link to a video showcasing the application -->
**Watch a quick demo of VisionCraft Studio in action:**

[![VisionCraft Studio Demo Video](https://img.shields.io/badge/Demo-Watch%20Video-red?style=for-the-badge&logo=youtube)](https://www.example.com/your_demo_video_link)

<!-- Or embed a GIF -->
<!-- <p align="center">
  <img src="link_to_your_demo.gif" alt="VisionCraft Studio Demo GIF" width="700"/>
</p> -->

## 📸 Screenshots

<!-- Add a few screenshots of your application -->
<p align="center">
  <b>Main Interface (Home Page)</b><br>
  <img src="https://i.ibb.co/8LPzdg90/vision-craft-home.png" alt="VisionCraft Studio Home Page" width="600"/><br><br>
  <b>Data Preparation Page</b><br>
  <img src="placeholder_screenshot_data.png" alt="VisionCraft Studio Data Page" width="600"/><br><br>
  <b>Training Configuration Page</b><br>
  <img src="placeholder_screenshot_train.png" alt="VisionCraft Studio Train Page" width="600"/><br><br>
  <b>Deployment & Live Inference Page</b><br>
  <img src="placeholder_screenshot_deploy.png" alt="VisionCraft Studio Deploy Page" width="600"/><br><br>
  <b>Dark Theme Example</b><br>
  <img src="placeholder_screenshot_dark_theme.png" alt="VisionCraft Studio Dark Theme" width="600"/>
</p>
*Note: Replace `placeholder_screenshot_*.png` with actual links to your screenshots.*

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **PyQt5:** For the Graphical User Interface.
* **Ultralytics YOLOv8:** For object detection and segmentation model training and inference.
* **OpenCV (cv2):** For image and video processing.
* **PyTorch:** Underlying framework for YOLOv8.
* **YAML:** For configuration files.
* **NumPy:** For numerical operations.

## ⚙️ Prerequisites

* Python 3.7 or higher.
* `pip` (Python package installer).
* Git (for cloning the repository).

The application will guide you to install `ultralytics` and other necessary model backends if they are not found.

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/vision-craft-studio.git](https://github.com/your-username/vision-craft-studio.git)
    cd vision-craft-studio
    ```
    *(Replace `your-username/vision-craft-studio` with your actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You might need to create a `requirements.txt` file. Based on `V6.py`, key packages are `PyQt5`, `opencv-python`, `torch`, `numpy`, `pyyaml`. `ultralytics` can be installed via the app or added to requirements.)*

    **Example `requirements.txt`:**
    ```
    PyQt5
    opencv-python
    numpy
    PyYAML
    torch
    torchvision 
    torchaudio
    ultralytics 
    # Add other specific versions if necessary
    ```

4.  **Ensure you have icons:**
    The application looks for icons in an `icons/` directory (with subdirectories for `light`/`dark` themes). Make sure this directory and the necessary icons are present. The script `V6.py` includes a function `create_dummy_icons_if_needed()` which can serve as a temporary placeholder or guide for required icons.

5.  **Run the application:**
    ```bash
    python V6.py
    ```

## 📖 Usage

VisionCraft Studio is organized into several tabs for a streamlined workflow:

1.  **🏠 Home:** Welcome screen with an overview of the application.
2.  **📊 Data:**
    * Click "Load Custom Dataset (.zip)" to select your dataset.
    * The dataset should be a zip file containing `images` and `labels` folders, and a `classes.txt` file (see YOLO dataset format).
    * The application will automatically process, split, and prepare the `data.yaml` file.
3.  **🧠 Train:**
    * Select a base YOLOv8 model (e.g., `yolov8s.pt`).
    * Set training parameters like epochs, image size.
    * Choose a project name for your training run.
    * Click "Start Training". The trained model (`best.pt`) will be saved in `trained_models_gui/`.
    * Optionally, auto-load the trained model into the Deploy tab.
4.  **🚀 Deploy:**
    * Load a trained model (e.g., your custom `best.pt` or other compatible model files).
    * **Live Inference:**
        * Select an available webcam and resolution.
        * Click "Start Webcam" to begin live detection/segmentation.
    * **File Inference:**
        * Click "Load Video/Image File" to run inference on a local media file.
    * **Model Conversion:**
        * Select an export format (e.g., ONNX, TFLite).
        * Click "Convert & Export Model". Exported models are saved in `exported_models_gui/`.
5.  **⚙️ Settings:**
    * Install or verify the Ultralytics library.
    * Select your preferred compute device (CPU or available CUDA GPUs).
    * Change the UI theme (Light/Dark).
6.  **ℹ️ About:** Information about VisionCraft Studio.

### Expected Dataset Structure

Your dataset should be provided as a `.zip` file with one of the following structures:

#### Structure 1: Pre-split Dataset
```
my_dataset.zip
├── images/
│   ├── train/              # Optional, can be flat
│   │   ├── img1.jpg
│   │   ├── img2.png
│   │   └── ...
│   ├── val/               # Optional, can be flat
│   │   ├── img3.jpg
│   │   └── ...
│   └── (Or all images directly here if not pre-split)
│       ├── img1.jpg
│       ├── img2.png
│       └── ...
├── labels/
│   ├── train/             # Mirrors images structure if pre-split
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   ├── val/              # Mirrors images structure if pre-split
│   │   ├── img3.txt
│   │   └── ...
│   └── (Or all labels directly here)
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── classes.txt
```

#### Structure 2: Simple Dataset
```
my_dataset.zip
├── dataset_root_folder/    # Optional intermediate folder
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── img1.txt
│   │   └── ...
│   └── classes.txt
└── (Or images/, labels/, classes.txt directly at the root of the zip)
```

**Note:**
- The `classes.txt` file should list one class name per line
- Label files (`.txt`) should be in YOLO format

### Project Structure

```
vision-craft-studio/
├── V6.py                     # Main application script
├── icons/                    # Directory for UI icons
│   ├── light/               # Icons for light theme
│   │   ├── home.png
│   │   └── ...
│   ├── dark/                # Icons for dark theme
│   │   ├── home.png
│   │   └── ...
│   └── app_icon.png         # Main application icon
├── yolo_gui_workspace/      # Workspace for dataset processing
│   └── dataset_processing/
├── trained_models_gui/      # Default directory for saved trained models
├── exported_models_gui/     # Default directory for exported/converted models
├── README.md                # This file
└── requirements.txt         # (Recommended) Python dependencies
```

## 🌊 Workflow

1.  **Prepare Data:** Use the "Data" tab to load and process your custom dataset.
2.  **Train Model:** Go to the "Train" tab, configure parameters, and train your YOLOv8 model.
3.  **Test & Deploy:**
    * Load your trained model in the "Deploy" tab.
    * Test with webcam or local files.
4.  **Export Model:** Convert your model to the desired format for deployment elsewhere.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please make sure to update tests as appropriate and adhere to a consistent coding style.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details (you'll need to create this file if you choose MIT).

## 🙏 Acknowledgements

* The [Ultralytics team](https://ultralytics.com/) for the YOLOv8 models and library.
* The [PyQt team](https://riverbankcomputing.com/software/pyqt/intro) for the GUI framework.
* [OpenCV](https://opencv.org/) for image processing capabilities.

## 📞 Contact

Midhun Mathew / Project Lead : midhun.ec2125@saintgits.org

Project Link: [https://github.com/memidhun/vcstudio](https://github.com/memidhun/vcstudio)

## ☕ Support the Project

If you find VisionCraft Studio helpful and would like to support its development, consider buying me a coffee! Your support helps maintain and improve this project.

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/memidhun)

Every coffee helps fuel more features and improvements! 🚀
