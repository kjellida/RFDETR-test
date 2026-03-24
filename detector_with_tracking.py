import cv2
import torch
import numpy as np
from rfdetr import RFDETRBase
from src.byte_tracker import BYTETracker 

#add resize if input resolution is large
def video_detection(
   input_path,
   output_path,
   conf_threshold=0.3,
   full_sweep_interval=10,
   max_rois=10,
   max_contour_area=30,
   display_padding= 1.15 
):


    # Check Cuda
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


    # Load model
    model = RFDETRBase()
    model.optimize_for_inference()


    # Video setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")


    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_size = (orig_w, orig_h)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Bytetrack setup 
    tracker = BYTETracker(track_thresh=0.4, track_buffer=30, match_thresh=0.8, fuse_score=False, frame_rate=fps)

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False
    )

    frame_index = 0

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame_index += 1


        # Motion detection
        fgMask = backSub.apply(frame)
        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)  #only needed if detect shadows = true in backsub
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) 
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        rois = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= max_contour_area]


        # Decide inference mode
        if len(rois) == 0 or len(rois) > max_rois or frame_index % full_sweep_interval == 0:
            run_full_frame = True
            mode_reason = "full-frame fallback"
        else:
            run_full_frame = False
            mode_reason = f"ROI-based, {len(rois)} blobs"


        boxes, scores = [], []


        # Full frame inference
        if run_full_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                detections = model.predict(rgb, threshold=conf_threshold)


            for box in detections.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(1.0)


        # ROI inference
        else:

            rois_np = np.array(rois)
            pad = 20 #maybe not pixels

            x1 = max(0, np.min(rois_np[:, 0]) - pad)
            y1 = max(0, np.min(rois_np[:, 1]) - pad)
            x2 = min(orig_w, np.max(rois_np[:, 0] + rois_np[:, 2]) + pad)
            y2 = min(orig_h, np.max(rois_np[:, 1] + rois_np[:, 3]) + pad)

            roi_w = x2 - x1
            roi_h = y2 - y1


            if roi_w > 0 and roi_h > 0:
                roi = frame[y1:y2, x1:x2]
                scale = 320 / max(roi_w, roi_h)  #unsure on sizing
                roi_small = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)))
                rgb_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)


                with torch.no_grad():
                    det = model.predict(rgb_roi, threshold=conf_threshold)


                for box in det.xyxy:
                    bx1, by1, bx2, by2 = map(int, box)
                    bx1 = int(bx1 / scale) + x1
                    by1 = int(by1 / scale) + y1
                    bx2 = int(bx2 / scale) + x1
                    by2 = int(by2 / scale) + y1
                    boxes.append([bx1, by1, bx2-bx1, by2-by1])
                    scores.append(1.0)


        if boxes:
            dets_for_tracker = np.array([[x, y, x+w, y+h, s] for (x,y,w,h),s in zip(boxes,scores)], dtype=np.float32)
        else:
            dets_for_tracker = np.empty((0,5), dtype=np.float32)

        
        if dets_for_tracker.shape[0] > 0:
           targets = tracker.update(dets_for_tracker, [orig_h, orig_w], [orig_h, orig_w])
        else:
           targets = []



        #  Draw tracked boxes
        for t in targets:
            x, y, w, h = t.tlwh
            x2, y2 = x + w, y + h
            x_disp = max(0, int(x) - display_padding)
            y_disp = max(0, int(y) - display_padding)
            x2_disp = min(orig_w, int(x2) + display_padding)
            y2_disp = min(orig_h, int(y2) + display_padding)
            cv2.rectangle(frame, (x_disp, y_disp), (x2_disp, y2_disp), (0, 255, 0), 2)

        out.write(frame)


        # Log Info
        if frame_index % 1 == 0:
            print(f"Frame {frame_index}: Mode={mode_reason}, ROIs={len(rois)}, Boxes={len(boxes)}")


    cap.release()
    out.release()
    print("Done. Saved to:", output_path)