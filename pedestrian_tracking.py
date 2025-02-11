import torch
import cv2
from sort.sort import Sort  # SORT tracking algorithm
import numpy as np

# Initialize SORT tracker
tracker = Sort()

# Load YOLOv5 model (you can use the pre-trained model or your custom-trained one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video stream (replace 'video.mp4' with 0 for webcam)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLOv5 to detect pedestrians
    results = model(frame)

    # Filter for 'person' class only (COCO class ID 0)
    detections = []
    for pred in results.xyxy[0]:  # Each prediction
        if int(pred[5]) == 0:  # Class 0 is 'person'
            x1, y1, x2, y2, conf = map(int, pred[:5])  # Extract bounding box and confidence
            detections.append([x1, y1, x2, y2, conf])  # Add to detections list

    # Convert detections to numpy array for SORT
    detections = np.array(detections)

    # Update the SORT tracker with current frame's detections
    tracked_objects = tracker.update(detections)

    # Draw bounding boxes and IDs for tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Draw ID

    # Display the frame with tracked objects
    cv2.imshow('Pedestrian Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
