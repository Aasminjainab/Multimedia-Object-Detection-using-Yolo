from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    DEFAULT_MODEL: str = os.getenv('DEFAULT_MODEL', 'yolov8s.pt')
    
    # File configurations
    ALLOWED_IMAGE_TYPES: List[str] = ["jpg", "jpeg", "png"]
    ALLOWED_VIDEO_TYPES: List[str] = ["mp4", "mov", "avi"]
    
    # Video processing
    TEMP_DIR: str = os.getenv('TEMP_DIR', 'temp')
    VIDEO_OUTPUT_FORMAT: str = os.getenv('VIDEO_OUTPUT_FORMAT', 'mp4v')
    MAX_VIDEO_DURATION: int = int(os.getenv('MAX_VIDEO_DURATION', '300'))  # 5 minutes default
    
    # UI configurations
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.25'))
    BBOX_COLOR: tuple = tuple(map(int, os.getenv('BBOX_COLOR', '0,255,0').split(',')))
    FONT_SCALE: float = float(os.getenv('FONT_SCALE', '0.5'))
    FONT_THICKNESS: int = int(os.getenv('FONT_THICKNESS', '2'))
    
    # Cache settings
    CACHE_DIR: str = os.getenv('CACHE_DIR', '.cache')
    MAX_CACHE_SIZE: int = int(os.getenv('MAX_CACHE_SIZE', '1024'))  # MB
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Validate model exists
            if cls.DEFAULT_MODEL not in cls.AVAILABLE_MODELS:
                raise ValueError(f"Invalid default model: {cls.DEFAULT_MODEL}")
            
            # Validate directories exist or can be created
            os.makedirs(cls.TEMP_DIR, exist_ok=True)
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
