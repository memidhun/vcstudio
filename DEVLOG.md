# VisionCraft Studio – Dev Log

All notable changes, features, and development notes for VisionCraft Studio will be documented in this file.

---
## [v0.1.0] – Initial Public Release
**Date:** 2024-06-XX

### Major Features
- Intuitive PyQt5 GUI for YOLO model training and deployment
- Dataset management: load, split, and prepare custom datasets from .zip files
- Automatic `data.yaml` generation for YOLO training
- Model training interface for YOLOv8 (detection & segmentation)
- Real-time inference via webcam, video, or image files
- Model export to ONNX, TorchScript, CoreML, TFLite, OpenVINO, and more
- In-app dependency management (Ultralytics, ONNX Runtime, etc.)
- Light and Dark (Dracula) UI themes
- Integrated console log and status/progress indicators
- Project structure for organized runs, trained models, and exports

### Improvements
- User-friendly error handling and status messages
- Automatic detection and installation prompts for missing dependencies
- Modern, responsive UI with theme switching
- Support for multiple compute devices (CPU, CUDA)

### Known Issues / Limitations
- Only YOLOv8 models are enabled by default (future versions may add YOLOv9+)
- No built-in annotation tool (external tools like Label Studio, CVAT, or Roboflow recommended)
- Some advanced YOLO training options are not exposed in the GUI
- No built-in test suite yet

### Notes
- See `README.md` for setup, usage, and dataset preparation instructions
- Icons must be present in the `icons/` directory (see code for details)

---

## Future Plans
- Add support for YOLOv9, YOLOv10+ and new model architectures
- Integrated annotation tool for dataset creation
- More advanced training options (augmentation, hyperparameters)
- Model performance metrics and training visualization
- Plugin system for custom export formats or inference backends
- Automated test suite and CI integration
- Improved error reporting and troubleshooting tools

---

*This dev log will be updated with each new release and major development milestone.* 