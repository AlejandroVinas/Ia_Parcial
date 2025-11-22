#!/usr/bin/env python3
"""
detect_and_recognize.py
Pipeline completo para detección y reconocimiento clásico de cartas sobre tapete verde.
Salida: CSV con detecciones y carpeta debug/ con imágenes intermedias.

Uso:
    python detect_and_recognize.py --input test_images/ --templates templates/ --out results.csv

Requisitos:
    pip install opencv-python numpy pandas
"""
import os
import cv2
import numpy as np
import argparse
import glob
import pandas as pd

# ---------- Parámetros (ajustables) ----------
LOWER_GREEN = np.array([35, 40, 40])   # HSV inicial (calibrar con calibrate_hsv.py)
UPPER_GREEN = np.array([85, 255, 255])
WARP_W, WARP_H = 200, 300
AREA_MIN_RATIO = 0.0008   # umbral de área relativo (ajustar)
DEBUG_DIR = "debug"
CSV_COLUMNS = ['filename','card_id','value','value_score','suit','suit_score','area','x_min','y_min','x_max','y_max','status']

# ---------- Utilidades ----------
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, W=WARP_W, H=WARP_H):
    rect = order_points(pts)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W,H))
    return warped, M

def load_templates(folder):
    templates = {'values':{}, 'suits':{}}
    for kind in ['values','suits']:
        p = os.path.join(folder, kind)
        if not os.path.isdir(p):
            continue
        for f in glob.glob(os.path.join(p,"*.png")):
            name = os.path.splitext(os.path.basename(f))[0]
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            templates[kind][name] = img_bin
    return templates

def match_best(template_dict, roi_gray):
    # binarize roi
    _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    best_name, best_score = None, -1.0
    for name, tpl in template_dict.items():
        tpl_h, tpl_w = tpl.shape
        r_h, r_w = roi_bin.shape
        # adapt size: resize tpl if larger than roi
        if tpl_h > r_h or tpl_w > r_w:
            scale_h = r_h / tpl_h
            scale_w = r_w / tpl_w
            scale = min(scale_h, scale_w)
            newsize = (max(1,int(tpl_w*scale)), max(1,int(tpl_h*scale)))
            tpl_resized = cv2.resize(tpl, newsize, interpolation=cv2.INTER_AREA)
        else:
            tpl_resized = tpl
        try:
            res = cv2.matchTemplate(roi_bin, tpl_resized, cv2.TM_CCOEFF_NORMED)
        except Exception as e:
            continue
        _, maxval, _, _ = cv2.minMaxLoc(res)
        if maxval > best_score:
            best_score = maxval
            best_name = name
    return best_name, float(best_score)

def count_pips(warp_gray):
    # Detect blobs / componentes para contar pips en cuerpo de la carta
    # Preprocess
    h,w = warp_gray.shape
    body = warp_gray[int(h*0.18):int(h*0.9), int(w*0.12):int(w*0.88)]
    _, th = cv2.threshold(body, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # morphological opening to separate small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-clean, connectivity=8)
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        wbox = stats[i, cv2.CC_STAT_WIDTH]
        hbox = stats[i, cv2.CC_STAT_HEIGHT]
        # heuristic: pips tienen tamaño limitado (ajustar según resolución)
        if 20 < area < 3000 and 5 < wbox < 120 and 5 < hbox < 120:
            count += 1
    return count

# ---------- Pipeline ----------
def process_image(path, templates, out_debug=True):
    img = cv2.imread(path)
    if img is None:
        return []
    h_img,w_img = img.shape[:2]
    img_area = h_img * w_img

    # Preprocess
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_cards = cv2.bitwise_not(mask_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    img_debug = img.copy()
    card_idx = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN_RATIO * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) >= 4:
            if len(approx) > 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                pts = box.astype("float32")
            else:
                pts = approx.reshape(4,2).astype("float32")
            warped, M = four_point_transform(img, pts)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # extract corner ROIs based on the normalized warp
            roi_val = gray[10:70, 10:60]     # value area
            roi_suit = gray[60:120, 10:60]   # suit area below value
            # If corner empty (rotation), also check opposite corner (rot 180)
            if np.mean(roi_val) > 250: # almost white -> maybe rotated
                roi_val = gray[-70:-10, -60:-10]
                roi_suit = gray[-120:-60, -60:-10]

            val_name, val_score = match_best(templates.get('values', {}), roi_val)
            suit_name, suit_score = match_best(templates.get('suits', {}), roi_suit)

            # pips counting to help validate numeric cards
            pips = count_pips(gray)

            status = "ok"
            # basic validation heuristics
            if val_score < 0.45 and suit_score < 0.45:
                status = "low_confidence"
            else:
                # for numeric cards (2..10) cross-check count of pips
                try:
                    numeric = int(val_name) if val_name and val_name.isdigit() else None
                except:
                    numeric = None
                if numeric is not None:
                    # if pip count available and differs drastically, reduce confidence
                    if pips>0 and abs(pips - numeric) > 2:
                        status = "pip_mismatch"

            # compute bounding box in original image
            xs = pts[:,0]; ys = pts[:,1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cv2.polylines(img_debug, [pts.astype(int)], True, (0,255,0), 2)
            cx = int(xs.mean()); cy = int(ys.mean())
            text = f"{val_name or '?'} {suit_name or '?'} ({val_score:.2f},{suit_score:.2f})"
            cv2.putText(img_debug, text, (cx-60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            detections.append({
                'card_id': card_idx,
                'value': val_name if val_name else '',
                'value_score': val_score,
                'suit': suit_name if suit_name else '',
                'suit_score': suit_score,
                'area': area,
                'bbox': (x_min,y_min,x_max,y_max),
                'status': status
            })
            card_idx += 1

    # debug output
    if out_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base}_mask.png"), mask)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base}_debug.png"), img_debug)

    return detections

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Imagen / carpeta con imágenes")
    parser.add_argument("--templates", required=True, help="Carpeta templates/ con subfolders values/ suits/")
    parser.add_argument("--out", default="results.csv", help="CSV de salida")
    args = parser.parse_args()

    templates = load_templates(args.templates)
    paths = []
    if os.path.isdir(args.input):
        exts = ("*.jpg","*.png","*.jpeg")
        for e in exts:
            paths.extend(glob.glob(os.path.join(args.input,e)))
    else:
        paths = [args.input]

    rows = []
    for p in paths:
        dets = process_image(p, templates, out_debug=True)
        for d in dets:
            x_min,y_min,x_max,y_max = d['bbox']
            rows.append({
                'filename': os.path.basename(p),
                'card_id': d['card_id'],
                'value': d['value'],
                'value_score': d['value_score'],
                'suit': d['suit'],
                'suit_score': d['suit_score'],
                'area': d['area'],
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'status': d['status']
            })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(args.out, index=False)
    print(f"Procesadas {len(paths)} imágenes. Resultados guardados en {args.out}. Debug en {DEBUG_DIR}/")

if __name__ == "__main__":
    main()
