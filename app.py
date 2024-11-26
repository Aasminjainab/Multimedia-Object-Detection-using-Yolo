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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    success: bool
    image: Optional[np.ndarray] = None
    error_message: Optional[str] = None

class YOLOModel:
    """Class to handle YOLO model operations"""
    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        self.model_name = model_name  # Store model name
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name: str) -> Optional[YOLO]:
        """Load YOLO model with error handling"""
        try:
            return YOLO(model_name)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
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
    
    def process_video(self, input_path: str) -> Tuple[bool, str]:
        """Process video file and return path to processed video"""
        if not os.path.exists(input_path):
            return False, "Input video file not found"
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False, "Failed to open video file"
            
            # Generate unique output filename
            timestamp = int(time.time())
            output_filename = f"processed_{timestamp}.mp4"
            temp_output = os.path.join(Config.TEMP_DIR, f"temp_{output_filename}")
            final_output = os.path.join(Config.TEMP_DIR, output_filename)
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer with h264 codec
            if os.name == 'nt':  # Windows
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:  # Linux/Mac
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            out = cv2.VideoWriter(
                temp_output,
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.model.detect_objects(rgb_frame)
                
                if result.success:
                    processed_frame = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
                    out.write(processed_frame)
                else:
                    out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log progress every 30 frames
                    logger.info(f"Processed {frame_count} frames")
            
            # Release video resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Convert to browser-compatible format using ffmpeg
            try:
                # Construct ffmpeg command
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', temp_output,  # Input file
                    '-c:v', 'libx264',  # Video codec
                    '-preset', 'medium',  # Encoding speed preset
                    '-movflags', '+faststart',  # Enable fast start for web playback
                    '-pix_fmt', 'yuv420p',  # Pixel format for maximum compatibility
                    final_output  # Output file
                ]
                
                # Run ffmpeg command
                import subprocess
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg error: {stderr.decode()}")
                    return False, f"FFmpeg conversion failed: {stderr.decode()}"
                
                # Clean up temporary file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                
                return True, final_output
                
            except Exception as e:
                logger.error(f"Error during ffmpeg conversion: {e}")
                return False, f"Error during video conversion: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False, str(e)
        finally:
            # Ensure resources are released
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            cv2.destroyAllWindows()

def download_youtube_video(youtube_url: str) -> Optional[str]:
    """Download YouTube video and return path to downloaded file"""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(Config.TEMP_DIR, '%(title)s.%(ext)s')
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_path = os.path.join(Config.TEMP_DIR, f"{info['title']}.mp4")
            return video_path if os.path.exists(video_path) else None
            
    except Exception as e:
        logger.error(f"Failed to retrieve video from YouTube: {e}")
        return None

def main():
    """Main application function"""
    # Set page configuration
    st.set_page_config(
        page_title="YOLO Object Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("MULTIMEDIA OBJECT DETECTION USING YOLO")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    
    # Model selection with description
    st.subheader("Model Selection")
    model_choice = st.selectbox(
        "Select YOLO Model",
        options=Config.AVAILABLE_MODELS,
        index=Config.AVAILABLE_MODELS.index(Config.DEFAULT_MODEL),
        format_func=lambda x: f"{x} - {Config.YOLO_MODELS[x]}"
    )
    
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
    
    # Initialize model and processors
    try:
        if st.session_state['model'] is None or st.session_state['model'].model_name != model_choice:
            with st.spinner("Loading YOLO model..."):
                st.session_state['model'] = YOLOModel(model_choice)
        model = st.session_state['model']
        image_processor = ImageProcessor(model)
        video_processor = VideoProcessor(model)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return
    
    tabs = st.tabs(["Image Detection", "Video Detection"])
    
    with tabs[0]:
        st.header("Image Detection")
        input_choice = st.radio("Select Input Method", ["Upload", "URL"])
        
        if input_choice == "Upload":
            uploaded_image = st.file_uploader(
                "Upload Image", 
                type=Config.ALLOWED_IMAGE_TYPES,
                key="image_uploader"
            )
            if uploaded_image is not None:
                try:
                    with st.spinner("Processing image..."):
                        image = Image.open(uploaded_image)
                        result = image_processor.process_image(image)
                        if result.success:
                            st.image(result.image, caption="Processed Image", use_container_width=True)
                        else:
                            st.error(result.error_message)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        elif input_choice == "URL":
            image_url = st.text_input("Image URL", key="image_url")
            if image_url:
                try:
                    with st.spinner("Processing image from URL..."):
                        result = image_processor.process_image(image_url)
                        if result.success:
                            st.image(result.image, caption="Processed Image", use_container_width=True)
                        else:
                            st.error(result.error_message)
                except Exception as e:
                    st.error(f"Error processing image URL: {str(e)}")
    
    with tabs[1]:
        st.header("Video Detection")
        video_choice = st.radio("Select Input Method", ["Upload", "YouTube"])
        
        if video_choice == "Upload":
            uploaded_video = st.file_uploader(
                "Upload Local Video", 
                type=Config.ALLOWED_VIDEO_TYPES,
                key="video_uploader"
            )
            if uploaded_video is not None:
                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save uploaded video
                    status_text.text("Saving uploaded video...")
                    input_video_path = os.path.join(Config.TEMP_DIR, uploaded_video.name)
                    with open(input_video_path, "wb") as f:
                        f.write(uploaded_video.getvalue())
                    
                    # Process video
                    status_text.text("Processing video...")
                    progress_bar.progress(25)
                    
                    success, result = video_processor.process_video(input_video_path)
                    progress_bar.progress(75)
                    
                    if success:
                        status_text.text("Loading processed video...")
                        st.video(result)
                        status_text.text("Video processing complete!")
                        progress_bar.progress(100)
                    else:
                        st.error(f"Failed to process video: {result}")
                    
                    # Cleanup
                    if os.path.exists(input_video_path):
                        os.remove(input_video_path)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    # Clear status
                    if 'status_text' in locals():
                        status_text.empty()
                    if 'progress_bar' in locals():
                        progress_bar.empty()
        
        elif video_choice == "YouTube":
            video_url = st.text_input("YouTube Video URL", key="youtube_url")
            if video_url:
                try:
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Download video
                    status_text.text("Downloading YouTube video...")
                    progress_bar.progress(25)
                    
                    video_path = download_youtube_video(video_url)
                    if not video_path:
                        st.error("Failed to download YouTube video")
                        return
                    
                    # Process video
                    status_text.text("Processing video...")
                    progress_bar.progress(50)
                    
                    success, result = video_processor.process_video(video_path)
                    progress_bar.progress(75)
                    
                    if success:
                        status_text.text("Loading processed video...")
                        st.video(result)
                        status_text.text("Video processing complete!")
                        progress_bar.progress(100)
                    else:
                        st.error(f"Failed to process video: {result}")
                    
                    # Cleanup
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    
                except Exception as e:
                    st.error(f"Error processing YouTube video: {str(e)}")
                finally:
                    # Clear status
                    if 'status_text' in locals():
                        status_text.empty()
                    if 'progress_bar' in locals():
                        progress_bar.empty()

if __name__ == "__main__":
    main()