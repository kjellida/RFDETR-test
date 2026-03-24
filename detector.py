import cv2
import torch
import numpy as np
from rfdetr import RFDETRBase


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


        boxes_to_draw = []


        # Full frame inference
        if run_full_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                detections = model.predict(rgb, threshold=conf_threshold)


            for box in detections.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes_to_draw.append((x1, y1, x2, y2)) 


        # ROI inference
        else:
           # xs = [x for x, y, w, h in rois]
           # ys = [y for x, y, w, h in rois]
          #  xe = [x + w for x, y, w, h in rois]
           # ye = [y + h for x, y, w, h in rois]

            rois_np = np.array(rois)
            pad = 20 #maybe not pixels?


          #  x1_roi = max(0, min(xs))
          #  y1_roi = max(0, min(ys))
           # x2_roi = min(orig_w, max(xe))
          #  y2_roi = min(orig_h, max(ye))


          #  roi_w = x2_roi - x1_roi
          #  roi_h = y2_roi - y1_roi


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
                    boxes_to_draw.append((bx1, by1, bx2, by2)) 
                    


        # Draw boxes
        for (x1, y1, x2, y2) in boxes_to_draw:
            #display padding 
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1) * display_padding / 2, (y2 - y1) * display_padding / 2

            # apply padding and clamp to image bounds
            x1_p = int(max(0, cx - w))
            y1_p = int(max(0, cy - h))
            x2_p = int(min(frame.shape[1], cx + w))
            y2_p = int(min(frame.shape[0], cy + h))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        out.write(frame)


        # Log Info
        if frame_index % 1 == 0:
            print(f"Frame {frame_index}: Mode={mode_reason}, ROIs={len(rois)}, Boxes={len(boxes_to_draw)}")


    cap.release()
    out.release()
    print("Done. Saved to:", output_path)