import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from rfdetr import RFDETRBase

# ----------------------
# Step 1: Parse ground truth XMLs
# ----------------------
def load_ground_truth(label_dir):
    """
    Returns a dictionary:
    frame_name -> list of boxes [xmin, ymin, xmax, ymax]
    """
    gt_dict = {}
    for file in os.listdir(label_dir):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(label_dir, file))
            root = tree.getroot()
            boxes = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                boxes.append([
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text)
                ])
            # Match XML filename to corresponding frame name
            frame_name = file.replace(".xml", ".jpg")
            gt_dict[frame_name] = boxes
    return gt_dict

# ----------------------
# Step 2: IoU computation
# ----------------------
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1]) +
             (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)
    return inter / union if union > 0 else 0

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """
    Returns TP, FP, FN for one frame
    """
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        match_found = False
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if iou(pred, gt) >= iou_thresh:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# ----------------------
# Step 3: Video detection with RFDETR
# ----------------------
def video_detection_with_eval(
    input_path,
    labels_dir,
    conf_threshold=0.2,
    full_sweep_interval=10,
    max_rois=3,
    max_contour_area=30,
    infer_width=560,
    infer_height=560
):
    # CUDA info
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load model
    model = RFDETRBase()
    model.optimize_for_inference()

    # Load ground truth
    gt_dict = load_ground_truth(labels_dir)
    print(f"Loaded ground truth for {len(gt_dict)} frames.")

    # Video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_size = (orig_w, orig_h)

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    frame_index = 0

    # Metrics
    total_tp = total_fp = total_fn = 0
    iou_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        frame_name = f"frame{frame_index:04d}.jpg"  # Adjust to your naming
        gt_boxes = gt_dict.get(frame_name, [])

        # Motion detection
        fgMask = backSub.apply(frame)
        _, thresh = cv2.threshold(fgMask.copy(), 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= max_contour_area]

        # Decide inference mode
        if len(rois) == 0 or len(rois) > max_rois or frame_index % full_sweep_interval == 0:
            run_full_frame = True
        else:
            run_full_frame = False

        boxes_to_draw = []

        if run_full_frame:
            # Full-frame inference
            resized = cv2.resize(frame, (infer_width, infer_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                detections = model.predict(rgb, threshold=conf_threshold)

            scale_x = orig_w / infer_width
            scale_y = orig_h / infer_height
            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                boxes_to_draw.append([x1, y1, x2, y2])
        else:
            # ROI-based inference
            for (x, y, w, h) in rois[:max_rois]:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                scale = 320 / max(w, h)
                small_w, small_h = int(w*scale), int(h*scale)
                roi_small = cv2.resize(roi, (small_w, small_h))
                rgb_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    det = model.predict(rgb_roi, threshold=conf_threshold)
                for box in det.xyxy:
                    bx1, by1, bx2, by2 = box.astype(int)
                    bx1 = int(bx1 / scale) + x
                    by1 = int(by1 / scale) + y
                    bx2 = int(bx2 / scale) + x
                    by2 = int(by2 / scale) + y
                    boxes_to_draw.append([bx1, by1, bx2, by2])

        # Draw boxes (optional, if you want video output)
        for (x1, y1, x2, y2) in boxes_to_draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Evaluation
        tp, fp, fn = match_boxes(boxes_to_draw, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for pred in boxes_to_draw:
            for gt in gt_boxes:
                iou_list.append(iou(pred, gt))

        if frame_index % 30 == 0:
            print(f"Frame {frame_index}: TP={tp}, FP={fp}, FN={fn}, Boxes={len(boxes_to_draw)}")

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    avg_iou = sum(iou_list)/len(iou_list) if iou_list else 0

    print("\n==== Evaluation Results ====")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"Average IoU: {avg_iou:.3f}")

    cap.release()

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    video_path = "FBD-SV-2024/videos/test_video.mp4"
    labels_dir = "FBD-SV-2024/labels"  # path to XML ground truth files

    video_detection_with_eval(video_path, labels_dir)