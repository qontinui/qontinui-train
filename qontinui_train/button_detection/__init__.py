"""
ML Infrastructure for Button Detection

This module provides ML-based button detection capabilities:
- Custom CNN architectures
- YOLO-based object detection
- Training infrastructure
- Inference pipelines
"""

from qontinui_train.button_detection.inference import ButtonDetectorInference
from qontinui_train.button_detection.models.button_cnn import (
    ButtonCNN,
    create_button_cnn,
)
from qontinui_train.button_detection.models.button_yolo import (
    ButtonYOLO,
    create_button_yolo,
)
from qontinui_train.button_detection.train_button_detector import (
    ButtonDetectorTrainer,
    COCOButtonDataset,
)
from qontinui_train.button_detection.utils import (
    analyze_dataset,
    coco_to_yolo,
    create_yolo_data_yaml,
    split_dataset,
    visualize_annotations,
)

__version__ = "1.0.0"

__all__ = [
    "ButtonDetectorInference",
    "ButtonCNN",
    "create_button_cnn",
    "ButtonYOLO",
    "create_button_yolo",
    "ButtonDetectorTrainer",
    "COCOButtonDataset",
    "analyze_dataset",
    "coco_to_yolo",
    "create_yolo_data_yaml",
    "split_dataset",
    "visualize_annotations",
]
