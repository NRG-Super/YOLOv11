from ultralytics import YOLO #Ultralytics is where you get the YOLO11 model

#The following commands are in case you do not have these libraries installed
import cv2 #cv2 is a library which has functions for computer vision and image processing
from PIL import Image #Stands for Python Imaging Library, used for opening and editing various image formats
import torch

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

#load YOLO11 model file from current directory
model = YOLO("yolo11n.pt").to(device)
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. source=0 for webcam
source = "bus.png"  
path = f"pictures/inputs/{source}"
results = model.predict(source=path)  # Display preds. Accepts all YOLO predict arguments, can add show=true to show image
resultName = f"{source.split('.')[0]}_results.jpg"
outputPath = f"pictures/outputs/{resultName}"
# Process results list, places bounding boxes and confidence levels, and finally saves and opens results.jpg
i = 0
for result in results:
    i = i+1
    #result_cuda = result.cuda() # Move each result to GPU
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=outputPath)  # save to disk