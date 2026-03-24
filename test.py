import cv2
import torch
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES


#CONFIG

INPUT_PATH = ""
OUTPUT_PATH = ""


print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))



model = RFDETRNano()
model.optimize_for_inference()



cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open input video")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (orig_w, orig_h))

# Background subtractor for motion detection
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

frame_index = 0