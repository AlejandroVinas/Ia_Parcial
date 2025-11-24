#!/usr/bin/env python3
"""
calibrate_hsv.py
Muestra en pantalla la imagen y los valores HSV del píxel bajo el ratón, y permite seleccionar con click un ROI
para calcular el rango HSV (min,max) que puedes usar para segmentar el tapete verde.

Uso:
    python calibrate_hsv.py --image sample.jpg

Instrucciones:
    - Mueve el ratón para ver HSV del píxel.
    - Haz click izquierdo y arrastra para seleccionar ROI; la consola mostrará min/max HSV del ROI.
    - Usa esos valores en detect_and_recognize.py
"""
import cv2
import numpy as np
import argparse

refPt = []
cropping = False
img = None
clone = None

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, img, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(img, refPt[0], refPt[1], (0,255,0), 2)
        cv2.imshow("image", img)
        # compute HSV range
        (x1,y1) = refPt[0]; (x2,y2) = refPt[1]
        roi = clone[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hmin = int(hsv[:,:,0].min()); smin = int(hsv[:,:,1].min()); vmin = int(hsv[:,:,2].min())
        hmax = int(hsv[:,:,0].max()); smax = int(hsv[:,:,1].max()); vmax = int(hsv[:,:,2].max())
        print("ROI HSV range:")
        print(f"LOWER = np.array([{hmin}, {smin}, {vmin}])")
        print(f"UPPER = np.array([{hmax}, {smax}, {vmax}])")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Imagen para calibración")
    args = parser.parse_args()
    global img, clone
    img = cv2.imread(args.image)
    if img is None:
        print("No se pudo cargar la imagen.")
        return
    clone = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    print("Mueve el ratón para leer HSV del píxel (se mostrará en consola). Selecciona ROI arrastrando click izquierdo.")
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            img = clone.copy()
        elif key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
