from ultralytics import YOLO

# Load a pretrained model
model = YOLO('checkpoints/yolov8n.pt')

# Perform detection on an image
results = model('/Users/melkor/Desktop/demo_chain.png')

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        coordinates = box.xyxy[0]  # get box coordinates
        confidence = box.conf[0]   # get confidence score
        class_id = box.cls[0]      # get class id

import torch
import torch.nn as nn
from core.metaphors.base import StateClassifier, StateMapper

source_dim = 32
target_dim = 3
hidden_dim = 256

mapper = StateMapper(source_dim, target_dim, hidden_dim)
classifier = StateClassifier(source_dim, 1, hidden_dim)

state = torch.randn([8, 32])

print(mapper(state).shape)
print(classifier(state).shape)
