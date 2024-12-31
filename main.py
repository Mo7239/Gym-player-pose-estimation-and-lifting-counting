import os
import cv2
from ultralytics import YOLO

# video readring
video_path = r'D:\pycharm\gym pose estimation\videos and images\4921646-hd_1066_1920_25fps.mp4'
cap = cv2.VideoCapture(video_path)

# model reading
model_path = r'D:\pycharm\gym pose estimation\yolo11x-pose.pt'
model = YOLO(model_path)

# Get the original video width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# res saving
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'gym.mp4', fourcc, fps, (original_width, original_height))



# Initialize variables for lifting count and state management
lifting_count = 0
lifting_in_progress = False

lifiting_mode = ' '


# Function to draw text with background rectangle
def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 255), padding=5):
    """
    Draw text with a background rectangle in OpenCV.
    """
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = position
    cv2.rectangle(
        img,
        (x - padding, y - text_size[1] - padding),
        (x + text_size[0] + padding, y + padding),
        bg_color,
        -1,
    )
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness)

while cap.isOpened():
    ret , frame  = cap.read()
    if not ret:
        break

    results = model.predict(frame)[0]
    key_points_data = results.keypoints.data

    for i ,key_point in enumerate(key_points_data):
        if key_point.shape[0]>0:
            right_shoulder = key_point[6][:2]
            right_elbow = key_point[8][:2]
            right_wrist = key_point[10][:2]

            y_elbow = right_elbow[1]
            y_wrist = right_wrist[1]

            if not lifting_in_progress and y_wrist > y_elbow:
                lifting_in_progress = True
                lifiting_mode  = 'DOWN'

            elif lifting_in_progress and y_wrist < y_elbow:
                lifting_count +=1
                lifting_in_progress= False
                lifiting_mode = 'UP'

    # Display lifting count on the frame
    display_text = f"Lifting Count: {lifting_count}"
    lifiting_mode_text = f"Lifting Mode: {lifiting_mode}"
    draw_text_with_background(frame, display_text, (50, 430), scale=2, thickness=3, bg_color=(0, 0, 255), text_color=(255, 255, 255))
    draw_text_with_background(frame, lifiting_mode_text, (50, 360), scale=2, thickness=3, bg_color=(0, 0, 255), text_color=(255, 255, 255))
    detection = results.plot()

    out.write(detection)

    cv2.imshow('lifting system' ,detection)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print('Done')

