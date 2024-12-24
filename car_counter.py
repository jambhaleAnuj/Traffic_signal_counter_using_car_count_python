import cv2
import math
import cvzone
import numpy as np
from sort import Sort
from ultralytics import YOLO
import time

# Initialize video capture
video_path = "Media/cars2.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
yolo_model = YOLO("Weights/yolov8n.pt")

# Define class names
class_labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load region mask
region_mask = cv2.imread("Media/mask.png")

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define line limits for counting
count_line = [199, 363, 1208, 377]

# List to store counted IDs
counted_ids = []

# Threshold for reducing timer
car_count_threshold = 10
normal_red_timer = 60  # seconds
reduced_red_timer = 30  # seconds

# Initialize timers
red_light_timer = normal_red_timer
last_timer_update_time = time.time()

# Variables to track reductions and cooldowns
timer_reduced_message = ""
reduction_count = 0
cooldown_active = False
cooldown_timer_start = None
cooldown_duration = 10  # seconds

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    masked_frame = cv2.bitwise_and(frame, region_mask)
    # Perform object detection
    detection_results = yolo_model(masked_frame, stream=True)

    # Collect detections
    detection_array = np.empty((0, 5))

    for result in detection_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            confidence = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])
            class_name = class_labels[class_id]

            if class_name in ["car", "truck", "motorbike", "bus"] and confidence > 0.3:
                detection_entry = np.array([x1, y1, x2, y2, confidence])
                detection_array = np.vstack((detection_array, detection_entry))
    tracked_objects = tracker.update(detection_array)

    # Draw count line
    cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (0, 255, 0), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        width, height = x2 - x1, y2 - y1

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Calculate center of the box
        center_x, center_y = x1 + width // 2, y1 + height // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        if count_line[0] < center_x < count_line[2] and count_line[1] - 20 < center_y < count_line[1] + 20:
            if obj_id not in counted_ids:
                counted_ids.append(obj_id)
                cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (255, 0, 0), 2)

    # Adjust red light timer if car count exceeds threshold
    current_time = time.time()
    if len(counted_ids) > car_count_threshold:
        if not cooldown_active:
            if reduction_count < 2 and red_light_timer != reduced_red_timer:
                timer_reduced_message = f"Timer reduced by {normal_red_timer - reduced_red_timer} seconds"
                red_light_timer = reduced_red_timer
                reduction_count += 1
    else:
        timer_reduced_message = ""
        red_light_timer = normal_red_timer
        reduction_count = 0  # Reset reduction count when below threshold

    # Cooldown logic
    if cooldown_active:
        elapsed_cooldown_time = current_time - cooldown_timer_start
        if elapsed_cooldown_time >= cooldown_duration:
            cooldown_active = False

    if reduction_count >= 2 and not cooldown_active:
        cooldown_active = True
        cooldown_timer_start = current_time

    # Calculate remaining timer
    elapsed_time = int(current_time - last_timer_update_time)
    remaining_time = max(0, red_light_timer - elapsed_time)

    if remaining_time == 0:
        last_timer_update_time = current_time
        red_light_timer = normal_red_timer
        timer_reduced_message = ""  # Clear message on reset

    # Display count, timer, and reduction message
    cvzone.putTextRect(frame, f'COUNT: {len(counted_ids)}', (20, 50), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)
    cvzone.putTextRect(frame, f'TIMER: {remaining_time}s', (20, 100), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

    if timer_reduced_message:
        cvzone.putTextRect(frame, timer_reduced_message, (20, 150), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Car Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
