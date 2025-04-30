# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 03:50:28 2025

@author: boate
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ───── Global settings ───────────────────────────────────────────────────────
NUM_QUESTIONS       = 30
NUM_CHOICES         = 5                      # A–E
CHOICE_LABELS       = ['A','B','C','D','E']

# ROI relative offsets
ROI_X_FACTOR        = 0.10
ROI_Y_FACTOR        = 14.20
ROI_WIDTH_FACTOR    = 0.60
ROI_HEIGHT_FACTOR   = 16.20

# Adaptive threshold params
ADAPTIVE_BLOCK      = 21  # must be odd
ADAPTIVE_C          = 10

# Morphology & dilation
MORPH_KERNEL_SIZE   = (3, 3)
DILATION_ITERS      = 1

# Line removal params (fraction of ROI width/height)
LINE_REMOVE_RATIO   = 0.6

# Blob detection params (ratios of cell area)
BLOB_MIN_AREA_RATIO   = 0.01  # too small blobs discarded
BLOB_MAX_AREA_RATIO   = 0.5   # overly large blobs discarded
BLOB_MIN_CIRCULARITY  = 0.75
BLOB_MIN_INERTIA      = 0.5

# Fallback density threshold (fraction of cell pixels)
DENSITY_THRESH_RATIO  = 0.15

# ───── Helpers ────────────────────────────────────────────────────────────────
def deskew(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=200, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for (x1,y1,x2,y2) in lines[:,0]:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if -45 < angle < 45:
                angles.append(angle)
    if angles:
        m = np.median(angles)
        h, w = image.shape[:2]
        rot = cv2.getRotationMatrix2D((w//2, h//2), m, 1.0)
        return cv2.warpAffine(image, rot, (w,h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return image


def multi_scale_template_match(img, template, scales=np.linspace(0.8,1.2,15)):
    best_val, best_loc, best_scale = -1, None, 1.0
    for s in scales:
        h, w = int(template.shape[0]*s), int(template.shape[1]*s)
        if h < 5 or w < 5: continue
        resized = cv2.resize(template, (w,h))
        res = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best_val:
            best_val, best_loc, best_scale = mx, ml, s
    return best_val, best_loc, best_scale


def extract_mc_answers(image_path, template_path, debug=False):
    # 1. Load & deskew
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Template match
    tpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"Cannot open template {template_path}")
    val, (mx, my), scale = multi_scale_template_match(gray, tpl)
    if val < 0.5:
        raise RuntimeError("Template match too weak; cannot find answer block.")

    # 3. Crop ROI
    th, tw = int(tpl.shape[0]*scale), int(tpl.shape[1]*scale)
    roi_x = mx + int(tw*ROI_X_FACTOR)
    roi_y = my + int(th*ROI_Y_FACTOR)
    roi_w = int(tw*ROI_WIDTH_FACTOR)
    roi_h = int(th*ROI_HEIGHT_FACTOR)
    h_img, w_img = gray.shape[:2]
    if roi_x<0 or roi_y<0 or roi_x+roi_w> w_img or roi_y+roi_h> h_img:
        raise RuntimeError("ROI out of bounds; adjust ROI factors.")
    roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # 4. Adaptive threshold + morphology
    clean = cv2.adaptiveThreshold(roi, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,
                                  ADAPTIVE_BLOCK, ADAPTIVE_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    # Dilation to reinforce marks
    clean = cv2.dilate(clean, kernel, iterations=DILATION_ITERS)

    # 5. Remove long lines (horizontal/vertical)
    # Horizontal
    hor_k_len = max(1, int(roi_w * LINE_REMOVE_RATIO))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_k_len, 1))
    hor_lines = cv2.morphologyEx(clean, cv2.MORPH_OPEN, hor_kernel)
    clean = cv2.subtract(clean, hor_lines)
    # Vertical
    ver_k_len = max(1, int(roi_h * LINE_REMOVE_RATIO))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_k_len))
    ver_lines = cv2.morphologyEx(clean, cv2.MORPH_OPEN, ver_kernel)
    clean = cv2.subtract(clean, ver_lines)

    # 6. Blob detection
    cell_h = roi_h / NUM_QUESTIONS
    cell_w = roi_w / NUM_CHOICES
    cell_area = cell_h * cell_w
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = BLOB_MIN_AREA_RATIO * cell_area
    params.maxArea = BLOB_MAX_AREA_RATIO * cell_area
    params.filterByCircularity = True
    params.minCircularity = BLOB_MIN_CIRCULARITY
    params.filterByInertia = True
    params.minInertiaRatio = BLOB_MIN_INERTIA
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(clean)

    # 7. Assign & fallback
    blobs_map = {i: [] for i in range(NUM_QUESTIONS)}
    for kp in keypoints:
        qi = min(NUM_QUESTIONS-1, int(kp.pt[1] / cell_h))
        ci = min(NUM_CHOICES-1, int(kp.pt[0] / cell_w))
        blobs_map[qi].append(ci)

    answers, centers = {}, []
    for qi in range(NUM_QUESTIONS):
        if blobs_map[qi]:
            best = int(np.argmax(np.bincount(blobs_map[qi], minlength=NUM_CHOICES)))
            answers[qi+1] = CHOICE_LABELS[best]
            centers.append((int(roi_x + (best+0.5)*cell_w), int(roi_y + (qi+0.5)*cell_h)))
        else:
            y1, y2 = int(qi*cell_h), int((qi+1)*cell_h)
            densities = [cv2.countNonZero(clean[y1:y2, int(ci*cell_w):int((ci+1)*cell_w)])
                         for ci in range(NUM_CHOICES)]
            thresh = DENSITY_THRESH_RATIO * cell_area
            best = int(np.argmax(densities))
            answers[qi+1] = CHOICE_LABELS[best] if densities[best]>=thresh else None
            if answers[qi+1]: centers.append((int(roi_x + (best+0.5)*cell_w), 
                                              int(roi_y + (qi+0.5)*cell_h)))

    # 8. Debug viz
    if debug:
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(disp, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0,255,0), 2)
        for kp in keypoints:
            x, y = int(kp.pt[0]+roi_x), int(kp.pt[1]+roi_y)
            cv2.circle(disp, (x,y), int(kp.size/2), (255,0,0), 2)
        for c in centers:
            cv2.circle(disp, c, 10, (0,0,255), 2)
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.title('Detections'); plt.axis('off'); plt.imshow(disp)
        plt.subplot(1,2,2); plt.title('Processed ROI'); plt.axis('off'); plt.imshow(clean, cmap='gray')
        plt.show()

    return answers, clean, centers

if __name__ == '__main__':
    ans, _, _ = extract_mc_answers('0M.jpg', 'TM.jpg', debug=True)
    print("Detected answers:")
    for q,a in ans.items(): print(f" Q{q:2d}: {a or '-'}")
