import cv2
from ultralytics import YOLO

# Open the video file
vid = cv2.VideoCapture(r"C:\Users\KIRUBA\Downloads\VID-20240923-WA0034.mp4")

# Load the YOLO model
model = YOLO(r"C:\Users\KIRUBA\Documents\yolo images\runs\detect\train13\weights\best.pt")

while True:
    ret, frame = vid.read()
    
    if not ret:
        break
    
    # Run inference
    results = model.predict(source=frame)

    # Iterate through results
    for result in results:
        # Access bounding boxes, confidence, and classes
        boxes = result.boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
        scores = result.boxes.conf   # Confidence scores
        classes = result.boxes.cls    # Class indices
        
        # Iterate through detected boxes
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.35:  # Set a confidence threshold (e.g., 0.5)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
                
                # Optionally, you can add the class label
                label = f'Class: {int(cls)}, Score: {score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Eagle Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
vid.release()
cv2.destroyAllWindows()
