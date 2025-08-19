import argparse
import time
import cv2
import math
import cvzone
import numpy as np
import yaml
from sort import Sort
from ultralytics import YOLO


def load_config(path: str | None) -> dict:
    default_cfg = {
        "video_path": "Media/cars2.mp4",
        "mask_path": "Media/mask.png",
        "weights_path": "Weights/yolov8n.pt",
        "classes_to_count": ["car", "truck", "motorbike", "bus"],
        "confidence_threshold": 0.3,
        "count_line": [199, 363, 1208, 377],
        "tracker": {"max_age": 20, "min_hits": 3, "iou_threshold": 0.3},
        "signal": {
            "car_count_threshold": 10,
            "normal_red_timer": 60,
            "reduced_red_timer": 30,
            "cooldown_duration": 10,
        },
    }
    if not path:
        return default_cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # merge shallow
        cfg = default_cfg | user_cfg
        # nested merges
        if "tracker" in user_cfg:
            cfg["tracker"] = default_cfg["tracker"] | user_cfg["tracker"]
        if "signal" in user_cfg:
            cfg["signal"] = default_cfg["signal"] | user_cfg["signal"]
        return cfg
    except FileNotFoundError:
        return default_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATSM Vehicle Counter (YOLOv8 + SORT)")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    p.add_argument("--video", dest="video_path", type=str, help="Override: video path")
    p.add_argument("--mask", dest="mask_path", type=str, help="Override: mask path")
    p.add_argument("--weights", dest="weights_path", type=str, help="Override: YOLO weights path")
    p.add_argument("--line", nargs=4, type=int, metavar=("x1", "y1", "x2", "y2"), help="Override: count line coords")
    p.add_argument("--threshold", type=int, dest="car_count_threshold", help="Override: congestion threshold")
    p.add_argument("--normal", type=int, dest="normal_red_timer", help="Override: normal red timer (s)")
    p.add_argument("--reduced", type=int, dest="reduced_red_timer", help="Override: reduced red timer (s)")
    p.add_argument("--cooldown", type=int, dest="cooldown_duration", help="Override: cooldown duration (s)")
    p.add_argument("--conf", type=float, dest="confidence_threshold", help="Override: detection confidence threshold")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides if provided
    if args.video_path: cfg["video_path"] = args.video_path
    if args.mask_path: cfg["mask_path"] = args.mask_path
    if args.weights_path: cfg["weights_path"] = args.weights_path
    if args.line: cfg["count_line"] = list(args.line)
    if args.car_count_threshold is not None: cfg["signal"]["car_count_threshold"] = args.car_count_threshold
    if args.normal_red_timer is not None: cfg["signal"]["normal_red_timer"] = args.normal_red_timer
    if args.reduced_red_timer is not None: cfg["signal"]["reduced_red_timer"] = args.reduced_red_timer
    if args.cooldown_duration is not None: cfg["signal"]["cooldown_duration"] = args.cooldown_duration
    if args.confidence_threshold is not None: cfg["confidence_threshold"] = args.confidence_threshold

    # Initialize video capture
    video_path = cfg["video_path"]
    cap = cv2.VideoCapture(video_path)

    # Load YOLO model with custom weights
    yolo_model = YOLO(cfg["weights_path"])

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
    region_mask = cv2.imread(cfg["mask_path"]) if cfg.get("mask_path") else None

    # Initialize tracker
    tracker_params = cfg.get("tracker", {})
    tracker = Sort(
        max_age=int(tracker_params.get("max_age", 20)),
        min_hits=int(tracker_params.get("min_hits", 3)),
        iou_threshold=float(tracker_params.get("iou_threshold", 0.3)),
    )

    # Define line limits for counting
    count_line = cfg["count_line"]

    # Set of counted IDs (use set for faster lookup)
    counted_ids = set()

    # Detection filter
    classes_to_count = set(cfg.get("classes_to_count", ["car", "truck", "motorbike", "bus"]))
    conf_thresh = float(cfg.get("confidence_threshold", 0.3))

    # Signal thresholds
    sig = cfg.get("signal", {})
    car_count_threshold = int(sig.get("car_count_threshold", 10))
    normal_red_timer = int(sig.get("normal_red_timer", 60))
    reduced_red_timer = int(sig.get("reduced_red_timer", 30))
    cooldown_duration = int(sig.get("cooldown_duration", 10))

    # Initialize timers
    red_light_timer = normal_red_timer
    last_timer_update_time = time.time()

    # Variables to track reductions and cooldowns
    timer_reduced_message = ""
    reduction_count = 0
    cooldown_active = False
    cooldown_timer_start = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if region_mask is not None:
            masked_frame = cv2.bitwise_and(frame, region_mask)
        else:
            masked_frame = frame

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

                if class_name in classes_to_count and confidence > conf_thresh:
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
                    counted_ids.add(obj_id)
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
            elapsed_cooldown_time = current_time - (cooldown_timer_start or current_time)
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
            counted_ids.clear()  # reset count per cycle

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


if __name__ == "__main__":
    main()
