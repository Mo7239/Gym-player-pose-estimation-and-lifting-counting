# Gym Pose Estimation - Exercise Counter
A computer vision application that uses YOLO pose estimation to track and count lifting exercises in real-time. The system detects key body points and analyzes arm movements to count repetitions during workout sessions.
Features

## Real-time pose detection and exercise counting
Visual feedback showing current lifting state (UP/DOWN)
On-screen counter for exercise repetitions
Support for video file processing
Custom visualization with status overlay

## Requirements

Python 3.8+
OpenCV (cv2)
Ultralytics YOLO
CUDA-capable GPU (recommended for optimal performance)

## Installation

Clone this repository:
Install the required packages:
```bash 
pip install opencv-python
pip install ultralytics
```
Download the YOLO pose estimation model (yolo11x-pose.pt) and place it in the project directory.


## Usage

Update the video path in the script to your input video:
```python
video_path = 'path/to/your/video.mp4'
```
Update the model path to point to your YOLO model:
```python
model_path = 'path/to/your/yolo11x-pose.pt'
```
Run the script:

```bash
python main.py
```
Press 'q' to exit the application.

## How It Works
The application:

Loads a pre-trained YOLO pose estimation model
Processes video frames in real-time
Detects key body points (shoulders, elbows, wrists)
Analyzes the relative positions of these points to determine lifting state
Counts completed repetitions when a full UP/DOWN cycle is detected
Displays real-time feedback with exercise count and current lifting state

## Output
The program generates:

A processed video file (gym.mp4) with pose detection overlay
Real-time visualization window showing:

Pose detection skeleton
Current repetition count
Lifting state (UP/DOWN)



Customization
You can modify the visualization parameters in the draw_text_with_background function:
```python
Copydraw_text_with_background(
    frame,
    display_text,
    position=(50, 430),
    scale=2,
    thickness=3,
    bg_color=(0, 0, 255),
    text_color=(255, 255, 255)
)
```
Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.
License


## Acknowledgments

Ultralytics for the YOLO implementation
OpenCV for the computer vision tools

## Contact
