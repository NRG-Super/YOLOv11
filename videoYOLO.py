from ultralytics import YOLO
import cv2
import torch

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLO11 model
model = YOLO("yolo11n.pt").to(device)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read one frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection on the frame
    results = model.predict(
        source=frame,
        device=device,
        verbose=False
    )

    # Draw YOLO detections on the frame
    annotated_frame = results[0].plot()

    # Display the video stream
    cv2.imshow("YOLO11 Live Detection - Press Q to Quit", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()