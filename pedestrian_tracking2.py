import torch
import cv2
from sort.sort import Sort  # SORT tracking algorithm
import numpy as np

# Initialize SORT tracker
tracker = Sort()

# Load YOLOv5 model (you can use the pre-trained model or your custom-trained one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video stream (replace 'video.mp4' with your video file path)
cap = cv2.VideoCapture('video.mp4')

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output
out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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

    # Draw bounding boxes for tracked objects (without IDs)
    for obj in tracked_objects:
        x1, y1, x2, y2, _ = obj.astype(int)  # Ignore obj_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    # Write the frame into the output video file
    out.write(frame)

    # Display the frame with tracked objects
    cv2.imshow('Pedestrian Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
