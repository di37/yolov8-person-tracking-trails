# Import OpenCV library for computer vision operations
import cv2
# Import YOLO class from Ultralytics library for YOLOv8 object detection and tracking
from ultralytics import YOLO
# Import collections for advanced data structures
# defaultdict: automatically creates missing keys with default values
# deque: double-ended queue with maximum length for efficient trail storage
from collections import defaultdict, deque

# Create a YOLO model instance by loading YOLOv8 nano model weights
model = YOLO("yolov8n.pt")

# Initialize video capture from a video file showing people walking
cap = cv2.VideoCapture("data/people_walking.mp4")

# Video writer will be initialized lazily after the first processed frame
out = None
# Try to read FPS from the source; fall back to 30.0 if unavailable
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Dictionary to map YOLO's internal object IDs to our custom sequential IDs
# This helps maintain consistent numbering even when YOLO IDs change
id_map = {}
# Counter for assigning new sequential IDs starting from 1
next_id = 1

# defaultdict with deque: each person gets a trail (deque) that stores up to 30 recent positions
# maxlen=30 automatically removes old positions when new ones are added
trail = defaultdict(lambda: deque(maxlen=30))
# Track how many consecutive frames each object has appeared
# Used to filter out brief false detections
appear = defaultdict(int)

# Start infinite loop for video processing
while True:
    # Read the next frame from the video file
    ret, frame = cap.read()
    # If no more frames available (end of video), exit the loop
    if not ret:
        break
    
    # Run YOLO object detection and tracking
    # persist=True: maintains object IDs across frames
    # classes=[0]: only detect "person" class from COCO dataset
    # verbose=False: suppress detailed output messages
    results = model.track(frame, persist=True, classes=[0], verbose=False)

    # Get the default annotated frame from YOLO
    annotated_frame = results[0].plot()

    # Initialize the video writer once we know the frame size
    if out is None:
        height, width = annotated_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Save as .mp4
        out = cv2.VideoWriter("data/people_walking_output.mp4", fourcc, float(fps), (width, height))

    # Check if any people were detected and have tracking IDs
    if results[0].boxes.id is not None:
        # Extract bounding box coordinates as NumPy array
        boxes = results[0].boxes.xyxy.numpy()
        # Extract tracking IDs as NumPy array
        ids = results[0].boxes.id.numpy()

        # Process each detected person (iterate through boxes and their corresponding IDs)
        for box, oid in zip(boxes, ids):
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box)
            # Calculate center point of the bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Increment appearance counter for this object ID
            appear[oid] += 1

            # Only assign a permanent ID if object has appeared for 5+ frames
            # This filters out brief false detections and ensures stable tracking
            if appear[oid] >= 5 and oid not in id_map:
                # Assign a new sequential ID to this object
                id_map[oid] = next_id
                # Increment for next new object
                next_id += 1

            # Only process objects that have been confirmed (appeared 5+ times)
            if oid in id_map:
                # Get the stable sequential ID for this object
                id = id_map[oid]
                # Add current center position to this person's trail
                trail[id].append((cx, cy))

                # Draw the bounding box around the person (red color)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw the ID number above the bounding box
                cv2.putText(annotated_frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw a small circle at the center point of the person
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                # Draw the movement trail by connecting consecutive points
                # Note: using oid instead of id here seems to be a bug in original code
                trail_points = list(trail[id])
                # Connect each point to the next with a red line
                for i in range(1, len(trail_points)):
                    cv2.line(annotated_frame, trail_points[i - 1], trail_points[i], (0, 0, 255), 2)

    # Display the annotated frame with trails and tracking information
    cv2.imshow("People with Trail", annotated_frame)

    # Write the processed frame to the output video
    if out is not None:
        out.write(annotated_frame)
    # Check if 'q' key was pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file handle
cap.release()
# Release the video writer if it was created
if out is not None:
    out.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
