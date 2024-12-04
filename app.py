import os
import cv2
import tempfile
import requests
import base64
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import streamlit as st
import yt_dlp as youtube_dl
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    success: bool
    image: Optional[np.ndarray] = None
    error_message: Optional[str] = None

@st.cache_resource
def load_yolo_model(model_name: str) -> YOLO:
    """Load YOLO model with caching"""
    try:
        if model_name not in Config.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        return YOLO(model_name)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

class YOLOModel:
    """Class to handle YOLO model operations"""
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        if not Config.validate_config():
            raise RuntimeError("Invalid configuration")
        self.model = load_yolo_model(model_name)
        
    def detect_objects(self, image: np.ndarray) -> DetectionResult:
        """Perform object detection on the input image"""
        if self.model is None:
            return DetectionResult(False, error_message="Model not loaded")
        
        try:
            results = self.model(image)
            annotated_image = image.copy()
            
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = self.model.names[int(result.cls)]
                confidence = result.conf.item()
                
                if confidence < Config.CONFIDENCE_THRESHOLD:
                    continue
                    
                cv2.rectangle(
                    annotated_image, 
                    (x1, y1), 
                    (x2, y2), 
                    Config.BBOX_COLOR, 
                    2
                )
                label_text = f'{label} {confidence:.2f}'
                cv2.putText(
                    annotated_image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    Config.FONT_SCALE,
                    Config.BBOX_COLOR,
                    Config.FONT_THICKNESS
                )
            
            return DetectionResult(True, annotated_image)
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return DetectionResult(False, error_message=str(e))

class ImageProcessor:
    """Class to handle image processing operations"""
    def __init__(self, model: YOLOModel):
        self.model = model
    
    def process_image(self, image: Union[Image.Image, str]) -> DetectionResult:
        """Process image from various sources (PIL Image or URL)"""
        try:
            if isinstance(image, str):
                image = self._load_image_from_url(image)
            
            if image is None:
                return DetectionResult(False, error_message="Failed to load image")
            
            # Convert image to RGB if it has an alpha channel
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            np_image = np.array(image)
            return self.model.detect_objects(np_image)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return DetectionResult(False, error_message=str(e))
    
    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL with support for base64"""
        try:
            if url.startswith('data:image'):
                header, encoded = url.split(',', 1)
                image_data = base64.b64decode(encoded)
                return Image.open(BytesIO(image_data))
            else:
                response = requests.get(url)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error loading image from URL: {e}")
            return None

class VideoProcessor:
    """Class to handle video processing operations"""
    def __init__(self, model: YOLOModel):
        self.model = model
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
    
    def process_video(self, input_path: str) -> Tuple[bool, Optional[str]]:
        """Process video file and return path to processed video"""
        cap = None
        writer = None
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return False, "Invalid video file"
                
            output_path = os.path.join(Config.TEMP_DIR, "processed_video.mp4")
            writer = self._setup_video_writer(cap, output_path)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                progress = min(100, int(frame_count * 100 / total_frames))
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                result = self.model.detect_objects(frame)
                if result.success:
                    writer.write(result.image)
            
            return True, output_path
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False, str(e)
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            progress_bar.empty()
            status_text.empty()
    
    def _setup_video_writer(self, cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
        """Set up video writer with input video properties"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_OUTPUT_FORMAT)
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def download_youtube_video(youtube_url: str) -> Optional[str]:
    """Download YouTube video and return path to downloaded file"""
    try:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, 'downloaded_video.mp4')
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_path
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path
    except Exception as e:
        logger.error(f"Failed to retrieve video from YouTube: {e}")
        return None

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in os.listdir(Config.TEMP_DIR):
            file_path = os.path.join(Config.TEMP_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {e}")

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """Validate image format and properties"""
    try:
        # Check image mode
        if image.mode not in ['RGB', 'RGBA']:
            return False, f"Unsupported image mode: {image.mode}"
        
        # Check image size
        max_dimension = 1920
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            return False, f"Image too large. Maximum dimension: {max_dimension}px"
        
        # Check if image is valid
        image.verify()
        return True, "Image is valid"
    except Exception as e:
        return False, str(e)

def main():
    """Main application function"""
    st.title("MULTIMEDIA OBJECT DETECTION USING YOLO")
    
    # Model selection with description
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Select YOLO Model",
        options=Config.AVAILABLE_MODELS,
        index=Config.AVAILABLE_MODELS.index(Config.DEFAULT_MODEL),
        format_func=lambda x: f"{x} - {Config.YOLO_MODELS[x]}"
    )
    
    # Initialize model using session state
    if 'model' not in st.session_state or st.session_state.get('model_choice') != model_choice:
        try:
            st.session_state.model = YOLOModel(model_choice)
            st.session_state.model_choice = model_choice
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    model = st.session_state.model
    image_processor = ImageProcessor(model)
    video_processor = VideoProcessor(model)
    
    # Display model capabilities
    model_type = "Detection"
    if "pose" in model_choice:
        model_type = "Pose Estimation"
        st.info("This model will detect and estimate human poses in the image/video.")
    elif "seg" in model_choice:
        model_type = "Instance Segmentation"
        st.info("This model will perform instance segmentation, creating precise masks for detected objects.")
    else:
        st.info("This model will detect and classify objects with bounding boxes.")
    
    tabs = st.tabs(["Image Detection", "Video Detection"])
    
    with tabs[0]:
        st.header("Image Detection")
        input_choice = st.radio("Select Input Method", ["Upload", "URL"])
        
        if input_choice == "Upload":
            uploaded_image = st.file_uploader(
                "Upload Image", 
                type=Config.ALLOWED_IMAGE_TYPES
            )
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                result = image_processor.process_image(image)
                if result.success:
                    st.image(result.image, caption="Processed Image", use_container_width=True)
                else:
                    st.error(result.error_message)
        
        elif input_choice == "URL":
            image_url = st.text_input("Image URL")
            if image_url:
                result = image_processor.process_image(image_url)
                if result.success:
                    st.image(result.image, caption="Processed Image", use_container_width=True)
                else:
                    st.error(result.error_message)
    
    with tabs[1]:
        st.header("Video Detection")
        video_choice = st.radio("Select Input Method", ["Upload", "YouTube"])
        
        if video_choice == "Upload":
            try:
                uploaded_video = st.file_uploader(
                    "Upload Local Video", 
                    type=Config.ALLOWED_VIDEO_TYPES
                )
                if uploaded_video is not None:
                    if uploaded_video.size > 200 * 1024 * 1024:  # 200MB limit
                        st.error("Video file is too large. Please upload a file smaller than 200MB.")
                        return
                        
                    input_video_path = os.path.join(Config.TEMP_DIR, uploaded_video.name)
                    with open(input_video_path, "wb") as f:
                        f.write(uploaded_video.read())
                    
                    try:
                        success, result = video_processor.process_video(input_video_path)
                        if success:
                            st.video(result)
                        else:
                            st.error(f"Error processing video: {result}")
                    finally:
                        cleanup_temp_files()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        elif video_choice == "YouTube":
            video_url = st.text_input("YouTube Video URL")
            if video_url:
                with st.spinner("Downloading video..."):
                    input_video_path = download_youtube_video(video_url)
                    if input_video_path:
                        try:
                            success, result = video_processor.process_video(input_video_path)
                            if success:
                                st.video(result)
                            else:
                                st.error(result)
                        finally:
                            cleanup_temp_files()
                    else:
                        st.error("Failed to download YouTube video")

if __name__ == "__main__":
    main()