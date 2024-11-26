from typing import List, Dict

class Config:
    # Model configurations with descriptions
    YOLO_MODELS = {
        "yolov8n.pt": "YOLOv8 Nano - Fastest and smallest model, best for CPU/edge devices",
        "yolov8s.pt": "YOLOv8 Small - Good balance of speed and accuracy",
        "yolov8m.pt": "YOLOv8 Medium - Better accuracy, still reasonable speed",
        "yolov8l.pt": "YOLOv8 Large - High accuracy, slower speed",
        "yolov8x.pt": "YOLOv8 XLarge - Highest accuracy, slowest speed",
        # Pose estimation models
        "yolov8n-pose.pt": "YOLOv8 Nano Pose - Fast pose estimation",
        "yolov8s-pose.pt": "YOLOv8 Small Pose - Balanced pose estimation",
        "yolov8m-pose.pt": "YOLOv8 Medium Pose - Accurate pose estimation",
        "yolov8l-pose.pt": "YOLOv8 Large Pose - High accuracy pose estimation",
        "yolov8x-pose.pt": "YOLOv8 XLarge Pose - Most accurate pose estimation",
        # Segmentation models
        "yolov8n-seg.pt": "YOLOv8 Nano Segmentation - Fast instance segmentation",
        "yolov8s-seg.pt": "YOLOv8 Small Segmentation - Balanced segmentation",
        "yolov8m-seg.pt": "YOLOv8 Medium Segmentation - Accurate segmentation",
        "yolov8l-seg.pt": "YOLOv8 Large Segmentation - High accuracy segmentation",
        "yolov8x-seg.pt": "YOLOv8 XLarge Segmentation - Most accurate segmentation"
    }
    
    AVAILABLE_MODELS: List[str] = list(YOLO_MODELS.keys())
    DEFAULT_MODEL: str = "yolov8s.pt"
    
    # File configurations
    ALLOWED_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png"]
    ALLOWED_VIDEO_TYPES: List[str] = ["mp4", "mov", "avi"]
    
    # Video processing
    TEMP_DIR: str = "temp"
    VIDEO_OUTPUT_FORMAT: str = "mp4v"
    
    # UI configurations
    CONFIDENCE_THRESHOLD: float = 0.25  # Lowered for better detection
    BBOX_COLOR: tuple = (0, 255, 0)
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 2
