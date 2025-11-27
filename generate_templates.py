#!/usr/bin/env python3
"""
generate_templates.py
Genera plantillas de valores y palos a partir de imágenes frontales (una carta por imagen,
o imágenes con una carta grande y centrada). Guarda recortes en templates/values y templates/suits.

Uso:
    python generate_templates.py --input templates_source/ --out templates/

Requisitos:
    pip install opencv-python numpy
"""
import os
import glob
import cv2
import numpy as np
import argparse

# default output
VALUES_OUT = "templates/values"
SUITS_OUT = "templates/suits"

def ensure_dirs():
    os.makedirs(VALUES_OUT, exist_ok=True)
    os.makedirs(SUITS_OUT, exist_ok=True)

def auto_crop_card(img):
    # Detect card by simple contour on high contrast background (assumes image mostly carta)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(255-th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) >= 4:
        if len(approx) > 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            pts = box.astype("float32")
        else:
            pts = approx.reshape(4,2).astype("float32")
        # reorder and warp
        rect = order_points(pts)
        W, H = 200, 300
        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (W,H))
        return warped
    else:
        # fallback to bounding rect
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        warp = cv2.resize(crop, (200,300))
        return warp

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_templates(warp, base_name):
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    # Value crop (top-left)
    val = gray[10:70, 10:60]
    suit = gray[60:120, 10:60]
    # also crop scaled versions
    val_resized = cv2.resize(val, (40,60), interpolation=cv2.INTER_AREA)
    suit_resized = cv2.resize(suit, (40,40), interpolation=cv2.INTER_AREA)
    # binarize
    _, vbin = cv2.threshold(val_resized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, sbin = cv2.threshold(suit_resized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return vbin, sbin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Carpeta con imágenes frontales de cartas")
    parser.add_argument("--out", default="templates", help="Carpeta destino")
    args = parser.parse_args()

    global VALUES_OUT, SUITS_OUT
    VALUES_OUT = os.path.join(args.out, "values")
    SUITS_OUT = os.path.join(args.out, "suits")
    ensure_dirs()

    imgs = []
    for ext in ("*.jpg","*.png","*.jpeg"):
        imgs.extend(glob.glob(os.path.join(args.input, ext)))

    for p in imgs:
        img = cv2.imread(p)
        base = os.path.splitext(os.path.basename(p))[0]
        warp = auto_crop_card(img)
        if warp is None:
            print(f"Skipping {p} (no card found)")
            continue
        v,s = extract_templates(warp, base)
        # Save with user-supplied base name; user can rename a posteriori si lo desea
        vpath = os.path.join(VALUES_OUT, f"{base}.png")
        spath = os.path.join(SUITS_OUT, f"{base}.png")
        cv2.imwrite(vpath, v)
        cv2.imwrite(spath, s)
        print(f"Saved templates for {base} -> {vpath}, {spath}")

if __name__ == "__main__":
    main()
