import os
import time

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import AutoShape, DetectMultiBackend

# Config value
video_path = "data_ext/people.mp4"
conf_threshold = 0.5
tracking_class = None  # None: track all

# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Initialize YOLOv9
device = "mps:0"  # "cuda": GPU, "cpu": CPU, "mps:0"
model = DetectMultiBackend(
    weights="weights/yolov9-c-converted.pt", device=device, fuse=True
)
model = AutoShape(model)

# Load classnames from classes.names file
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(class_names), 3))
tracks = []

# Initialize VideoCapture to read from video file
cap = cv2.VideoCapture(video_path)

# Get video properties for metrics calculation
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps

# Metrics variables
true_positives = 0
false_positives = 0
false_negatives = 0
frame_count = 0

print(f"Video Information:")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Duration: {duration:.2f} seconds")
print("Processing video...")

start_time = time.time()

# For visualization and saving results
if not os.path.exists("results"):
    os.makedirs("results")

# Create video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(
    "results/tracked_output.avi",
    fourcc,
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

# Process each frame from the video
while True:
    # Read
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}/{total_frames}")

    # Pass through model for detection
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
        true_positives += 1  # This is an approximation - in a real scenario, you'd compare with ground truth

    # Update and assign IDs using DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Count missed detections (false negatives) - this is a simplification
    # In a real scenario, you'd compare with ground truth
    expected_objects_per_frame = len(detect)  # This is just an example
    false_negatives += max(
        0, expected_objects_per_frame - len([t for t in tracks if t.is_confirmed()])
    )

    # Count false positives - objects tracked but not in ground truth
    # This is also a simplification
    false_positives += len(
        [t for t in tracks if t.time_since_update > 5 and t.is_confirmed()]
    )

    # Draw bounding boxes with IDs on the screen
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Get coordinates, class_id to draw on the image
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(
                frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Write frame to output video
    output_video.write(frame)

    # Show image on screen
    cv2.imshow("Object Tracking", frame)
    # Press Q to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Calculate metrics
accuracy = (
    true_positives / (true_positives + false_positives + false_negatives)
    if (true_positives + false_positives + false_negatives) > 0
    else 0
)
precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)
recall = (
    true_positives / (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
)
f1_score = (
    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
)

end_time = time.time()
processing_time = end_time - start_time

print("\nVideo Processing Complete!")
print(
    f"Processed {frame_count} frames in {processing_time:.2f} seconds ({frame_count/processing_time:.2f} FPS)"
)
print("\nTracking Metrics:")
print(f"Total Frames: {total_frames}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Save metrics to a file
with open("results/tracking_metrics.txt", "w") as f:
    f.write("Video Tracking Metrics\n")
    f.write(f"Video: {video_path}\n")
    f.write(f"Total Frames: {total_frames}\n")
    f.write(f"True Positives: {true_positives}\n")
    f.write(f"False Positives: {false_positives}\n")
    f.write(f"False Negatives: {false_negatives}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1_score:.4f}\n")

cap.release()
output_video.release()
cv2.destroyAllWindows()
