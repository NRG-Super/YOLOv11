import os
import ultralytics
from ultralytics import YOLO
from PIL import Image
import requests
import supervision as sv

HOME = os.getcwd()

model = YOLO('yolo11n.pt')
image = Image.open(requests.get('https://media.roboflow.com/notebooks/examples/dog.jpeg', stream=True).raw)
result = model.predict(image, conf=0.25)[0]

# detections = sv.Detections.from_ultralytics(result)

# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

# annotated_image = image.copy()
# annotated_image = box_annotator.annotate(annotated_image, detections=detections)
# annotated_image = label_annotator.annotate(annotated_image, detections=detections)

# sv.plot_image(annotated_image, size=(10, 10))

from roboflow import Roboflow

#Code to download dataset from Roboflow, only need to run once

# Replace 'your_api_key' with your actual Roboflow API key
#ROBOFLOW_API_KEY = 'S89kqplRTMgZScICinVD'
#rf = Roboflow(api_key=ROBOFLOW_API_KEY)

#workspace = rf.workspace("liangdianzhong")
#project = workspace.project("-qvdww")
#version = project.version(3)
#dataset = version.download("yolov11")

# result = model.train(
#     data="C:/Users/Admin/Desktop/YOLOv11/fineTune/data.yaml",
#     epochs=20,
#     imgsz=640,
#     device=0,
#     batch=0.9,
#     workers=0,
#     plots=True
# )

import re
def get_highest_subfolder(path: str):
    items = os.listdir(path)
    directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    numeric_values = []
    for dir_name in directories:
        numbers = re.findall(r'\d+', dir_name)
        if numbers:
            numeric_values.extend(map(int, numbers))
    return max(numeric_values) if numeric_values else 0

highestNum = get_highest_subfolder("C:/Users/Admin/Desktop/YOLOv11/runs/detect/")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(f'C:/Users/Admin/Desktop/YOLOv11/runs/detect/train{highestNum}/confusion_matrix.png')
plt.imshow(img)
plt.axis('off') # Optional: hide the axis
plt.show()

img = mpimg.imread(f'C:/Users/Admin/Desktop/YOLOv11/runs/detect/train{highestNum}/results.png')
plt.imshow(img)
plt.axis('off') # Optional: hide the axis
plt.show()

img = mpimg.imread(f'C:/Users/Admin/Desktop/YOLOv11/runs/detect/train{highestNum}/val_batch0_pred.jpg')
plt.imshow(img)
plt.axis('off') # Optional: hide the axis
plt.show()

metrics = model.val(
    data="C:/Users/Admin/Desktop/YOLOv11/fineTune/data.yaml",
    device=0,
    batch=0.9,
)

#Shows tensors and device in use
#print(result.boxes.cls)