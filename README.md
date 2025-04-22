# Multimedia Object Detection using YOLO

This project is a Streamlit application that performs object detection in images and videos using YOLO (You Only Look Once) models. It supports various YOLO models, allowing users to choose between different sizes and capabilities, including object detection, pose estimation, and instance segmentation.

## Features

- **Object Detection in Images and Videos:** Detect objects in uploaded images, images from URLs, uploaded videos, and YouTube videos.
- **Multiple YOLO Models:** Supports a range of YOLOv8 models (Nano, Small, Medium, Large, XLarge) for object detection, pose estimation, and instance segmentation.
- **Streamlit UI:** User-friendly web interface built with Streamlit for easy interaction.
- **Configurable Settings:** Allows customization of confidence threshold, bounding box color, and more via `config.py`.
- **Temporary File Handling:** Manages temporary files for video processing and cleanup.

## Available Models

The application supports the following YOLO models, which can be selected from the UI:

- `yolov8n.pt`: YOLOv8 Nano - Fastest and smallest model, best for CPU/edge devices
- `yolov8s.pt`: YOLOv8 Small - Good balance of speed and accuracy (Default)
- `yolov8m.pt`: YOLOv8 Medium - Better accuracy, still reasonable speed
- `yolov8l.pt`: YOLOv8 Large - High accuracy, slower speed
- `yolov8x.pt`: YOLOv8 XLarge - Highest accuracy, slowest speed
- `yolov8n-pose.pt`: YOLOv8 Nano Pose - Fast pose estimation
- `yolov8s-pose.pt`: YOLOv8 Small Pose - Balanced pose estimation
- `yolov8m-pose.pt`: YOLOv8 Medium Pose - Accurate pose estimation
- `yolov8l-pose.pt`: YOLOv8 Large Pose - High accuracy pose estimation
- `yolov8x-pose.pt`: YOLOv8 XLarge Pose - Most accurate pose estimation
- `yolov8n-seg.pt`: YOLOv8 Nano Segmentation - Fast instance segmentation
- `yolov8s-seg.pt`: YOLOv8 Small Segmentation - Balanced segmentation
- `yolov8m-seg.pt`: YOLOv8 Medium Segmentation - Accurate segmentation
- `yolov8l-seg.pt`: YOLOv8 Large Segmentation - High accuracy segmentation
- `yolov8x-seg.pt`: YOLOv8 XLarge Segmentation - Most accurate segmentation

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd YOLO_TESTING_2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

- **Environment Variables:**
  - You can configure the application using environment variables. Create a `.env` file in the project root to override default settings.
  - Available environment variables are defined in `config.py`. Example `.env` file:
    ```
    DEFAULT_MODEL=yolov8m.pt
    CONFIDENCE_THRESHOLD=0.3
    BBOX_COLOR=255,0,0
    ```

## How to Run

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

Open your browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

## Usage

### Image Detection

1. Navigate to the "Image Detection" tab.
2. Choose input method:
   - **Upload:** Upload an image file from your local machine (supports JPG, JPEG, PNG).
   - **URL:** Enter the URL of an image.
3. Select a YOLO model from the dropdown.
4. The processed image with detected objects will be displayed.

### Video Detection

1. Navigate to the "Video Detection" tab.
2. Choose input method:
   - **Upload:** Upload a video file from your local machine (supports MP4, MOV, AVI, maximum 200MB).
   - **YouTube:** Enter a YouTube video URL.
3. Select a YOLO model from the dropdown.
4. For video upload, processing will start, and a progress bar will be shown. For YouTube videos, the video will be downloaded first, then processed.
5. The processed video with detected objects will be displayed.

## Project Structure

```
YOLO_TESTING_2/
├── .env               # Environment configuration file (optional)
├── .gitattributes     # Git attributes configuration
├── app.py             # Main Streamlit application file
├── config.py          # Configuration settings
├── README.md          # Project README file
├── requirements.txt   # Project dependencies
└── temp/              # Temporary directory for video processing
```

---

**Note:** Ensure you have FFmpeg installed on your system for video processing to work correctly. For YouTube video downloads, `yt-dlp` is used, make sure it is installed and up to date.
