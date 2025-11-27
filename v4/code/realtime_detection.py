#!/usr/bin/env python3
"""
realtime_detection.py - VERSIÓN CORREGIDA
Detección y reconocimiento de cartas en TIEMPO REAL usando webcam externa.

Uso:
    python realtime_detection.py --templates ../templates/ --camera 1

Controles:
    ESPACIO: Capturar foto
    C: Calibrar verde (arrastra ratón)
    R: Reset calibración
    Q/ESC: Salir
"""

import cv2
import numpy as np
import argparse
import os
import time
import glob
from datetime import datetime
import pandas as pd

# ========== PARÁMETROS GLOBALES ==========
# Calibración HSV inicial (ajustar según tu tapete)
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Configuración
WARP_W, WARP_H = 200, 300
AREA_MIN_RATIO = 0.0008
MIN_VALUE_SCORE = 0.40
MIN_SUIT_SCORE = 0.40

# Directorios
CAPTURE_DIR = "captures"
RESULTS_CSV = "captures/captures_log.csv"

# Estado
calibration_mode = False
calibration_points = []


# ========== FUNCIONES DE GEOMETRÍA ==========

def order_points(pts):
    """Ordena 4 puntos: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, W=WARP_W, H=WARP_H):
    """Transformación de perspectiva"""
    rect = order_points(pts)
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped, M


# ========== CARGA DE PLANTILLAS ==========

def load_templates(folder):
    """Carga plantillas desde carpeta templates/"""
    templates = {'values': {}, 'suits': {}}
    
    for kind in ['values', 'suits']:
        path = os.path.join(folder, kind)
        if not os.path.isdir(path):
            print(f"ADVERTENCIA: No se encontró {path}")
            continue
        
        for filepath in glob.glob(os.path.join(path, "*.png")):
            name = os.path.splitext(os.path.basename(filepath))[0]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Binarizar plantilla
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            templates[kind][name] = img_bin
        
        print(f"  Cargadas {len(templates[kind])} plantillas de {kind}")
    
    return templates


# ========== MATCHING ==========

def match_best(template_dict, roi_gray):
    """Template matching con mejor score"""
    if roi_gray is None or roi_gray.size == 0:
        return None, 0.0
    
    # Binarizar ROI
    _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    best_name, best_score = None, -1.0
    
    for name, tpl in template_dict.items():
        tpl_h, tpl_w = tpl.shape
        r_h, r_w = roi_bin.shape
        
        # Redimensionar plantilla si es necesario
        if tpl_h > r_h or tpl_w > r_w:
            scale = min(r_h / tpl_h, r_w / tpl_w) * 0.9
            newsize = (max(1, int(tpl_w * scale)), max(1, int(tpl_h * scale)))
            tpl_resized = cv2.resize(tpl, newsize, interpolation=cv2.INTER_AREA)
        else:
            tpl_resized = tpl
        
        try:
            res = cv2.matchTemplate(roi_bin, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, _ = cv2.minMaxLoc(res)
            
            if maxval > best_score:
                best_score = maxval
                best_name = name
        except:
            continue
    
    return best_name, float(best_score)


# ========== DETECCIÓN EN TIEMPO REAL ==========

def preprocess_frame(frame, lower_hsv, upper_hsv):
    """Preprocesa frame para segmentación"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_cards = cv2.bitwise_not(mask_green)
    
    # Morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def detect_cards_realtime(frame, templates, lower_hsv, upper_hsv, draw=True):
    """Detecta y reconoce cartas en frame"""
    h_img, w_img = frame.shape[:2]
    img_area = h_img * w_img
    
    # Preprocesar
    mask = preprocess_frame(frame, lower_hsv, upper_hsv)
    
    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    annotated = frame.copy() if draw else None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtrar por área
        if area < AREA_MIN_RATIO * img_area:
            continue
        
        # Aproximación poligonal
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) >= 4:
            # Obtener 4 puntos
            if len(approx) > 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                pts = box.astype("float32")
            else:
                pts = approx.reshape(4, 2).astype("float32")
            
            # Transformación de perspectiva
            try:
                warped, M = four_point_transform(frame, pts)
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                continue
            
            # Extraer ROIs
            roi_val = gray[10:70, 10:60]
            roi_suit = gray[60:120, 10:60]
            
            # Verificar rotación
            if np.mean(roi_val) > 240:
                roi_val = gray[-70:-10, -60:-10]
                roi_suit = gray[-120:-60, -60:-10]
            
            # Matching
            val_name, val_score = match_best(templates.get('values', {}), roi_val)
            suit_name, suit_score = match_best(templates.get('suits', {}), roi_suit)
            
            # Estado
            if val_score >= MIN_VALUE_SCORE and suit_score >= MIN_SUIT_SCORE:
                status = "OK"
                color = (0, 255, 0)
            else:
                status = "LOW_CONF"
                color = (0, 165, 255)
            
            # Guardar detección
            xs, ys = pts[:, 0], pts[:, 1]
            detection = {
                'value': val_name if val_name else '?',
                'suit': suit_name if suit_name else '?',
                'value_score': val_score,
                'suit_score': suit_score,
                'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                'status': status
            }
            detections.append(detection)
            
            # Dibujar
            if draw:
                cv2.polylines(annotated, [pts.astype(int)], True, color, 3)
                
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{val_name or '?'}-{suit_name or '?'}"
                
                # Fondo texto
                (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (cx - w_text//2 - 5, cy - h_text - 5), 
                             (cx + w_text//2 + 5, cy + 5), (0, 0, 0), -1)
                
                cv2.putText(annotated, text, (cx - w_text//2, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                score_text = f"{val_score:.2f}/{suit_score:.2f}"
                cv2.putText(annotated, score_text, (cx - w_text//2, cy + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return detections, annotated, mask


# ========== HUD ==========

def draw_hud(frame, detections, fps, calibrating=False):
    """Dibuja interfaz con información"""
    h, w = frame.shape[:2]
    
    # Panel superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    title = "CALIBRANDO TAPETE VERDE" if calibrating else "DETECTOR DE CARTAS - TIEMPO REAL"
    cv2.putText(frame, title, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if calibrating else (255, 255, 255), 2)
    
    info_text = f"FPS: {fps:.1f} | Cartas: {len(detections)}"
    cv2.putText(frame, info_text, (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Panel inferior
    controls_y = h - 100
    cv2.rectangle(frame, (0, controls_y), (w, h), (0, 0, 0), -1)
    
    controls = ["ESPACIO: Capturar", "C: Calibrar", "R: Reset", "Q: Salir"]
    for i, ctrl in enumerate(controls):
        x_pos = 20 + (i * (w // 4))
        cv2.putText(frame, ctrl, (x_pos, controls_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Detecciones recientes
    if detections and not calibrating:
        y_offset = controls_y - 20
        for i, det in enumerate(detections[-3:]):
            card_text = f"{i+1}. {det['value']}-{det['suit']} ({det['status']})"
            color = (0, 255, 0) if det['status'] == 'OK' else (0, 165, 255)
            cv2.putText(frame, card_text, (w - 250, y_offset - (i * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ========== CALIBRACIÓN ==========

def calibrate_green_interactive(frame, points):
    """Calibra verde desde ROI seleccionado"""
    if len(points) < 2:
        return None, None
    
    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None, None
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower = np.array([
        int(hsv_roi[:, :, 0].min()),
        int(hsv_roi[:, :, 1].min()),
        int(hsv_roi[:, :, 2].min())
    ])
    
    upper = np.array([
        int(hsv_roi[:, :, 0].max()),
        int(hsv_roi[:, :, 1].max()),
        int(hsv_roi[:, :, 2].max())
    ])
    
    return lower, upper


# ========== GUARDAR CAPTURAS ==========

def save_capture(frame, detections, capture_num):
    """Guarda captura y log CSV"""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{capture_num:03d}_{timestamp}.jpg"
    filepath = os.path.join(CAPTURE_DIR, filename)
    
    cv2.imwrite(filepath, frame)
    
    # CSV
    rows = []
    for i, det in enumerate(detections):
        rows.append({
            'timestamp': timestamp,
            'capture_num': capture_num,
            'filename': filename,
            'card_id': i,
            'value': det['value'],
            'value_score': det['value_score'],
            'suit': det['suit'],
            'suit_score': det['suit_score'],
            'status': det['status']
        })
    
    df = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)
    
    return filename


# ========== CALLBACK RATÓN ==========

def mouse_callback(event, x, y, flags, param):
    """Callback para calibración"""
    global calibration_points, calibration_mode
    
    if calibration_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP and len(calibration_points) == 1:
            calibration_points.append((x, y))


# ========== MAIN ==========

def main():
    global calibration_mode, calibration_points, LOWER_GREEN, UPPER_GREEN
    
    parser = argparse.ArgumentParser(description="Detección en tiempo real")
    parser.add_argument("--templates", required=True, help="Carpeta templates/")
    parser.add_argument("--camera", type=int, default=0, help="ID cámara")
    parser.add_argument("--width", type=int, default=1280, help="Ancho")
    parser.add_argument("--height", type=int, default=720, help="Alto")
    args = parser.parse_args()
    
    # Cargar plantillas
    print("=" * 60)
    print("CARGANDO PLANTILLAS")
    print("=" * 60)
    templates = load_templates(args.templates)
    
    if not templates['values'] or not templates['suits']:
        print("ERROR: No se cargaron plantillas")
        return
    
    # Abrir cámara
    print(f"\nAbriendo cámara {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir cámara {args.camera}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Cámara: {actual_width}x{actual_height}")
    
    # Ventana
    window_name = "Detector de Cartas - Tiempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("  ESPACIO: Capturar")
    print("  C: Calibrar verde")
    print("  R: Reset")
    print("  Q/ESC: Salir")
    print("=" * 60 + "\n")
    
    capture_count = 0
    fps = 0
    prev_time = time.time()
    
    # Ventana de máscara (debug)
    cv2.namedWindow("Mask (Debug)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask (Debug)", 640, 480)
    
    # Loop principal
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: No se pudo capturar frame")
            break
        
        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Modo calibración
        if calibration_mode:
            display_frame = frame.copy()
            
            if len(calibration_points) == 2:
                cv2.rectangle(display_frame, calibration_points[0], calibration_points[1], 
                             (0, 255, 0), 2)
            
            cv2.putText(display_frame, "ARRASTRA para seleccionar verde", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Presiona C cuando termines", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
        
        # Modo detección
        else:
            detections, annotated_frame, mask = detect_cards_realtime(
                frame, templates, LOWER_GREEN, UPPER_GREEN, draw=True
            )
            
            draw_hud(annotated_frame, detections, fps, calibrating=False)
            cv2.imshow(window_name, annotated_frame)
            
            # Mostrar máscara (debug)
            cv2.imshow("Mask (Debug)", mask)
        
        # Teclas
        key = cv2.waitKey(1) & 0xFF
        
        # ESPACIO: Capturar
        if key == ord(' ') and not calibration_mode:
            detections, annotated_frame, _ = detect_cards_realtime(
                frame, templates, LOWER_GREEN, UPPER_GREEN, draw=True
            )
            
            if detections:
                capture_count += 1
                filename = save_capture(annotated_frame, detections, capture_count)
                print(f"\n✓ Captura {capture_count}: {filename}")
                print(f"  Cartas: {len(detections)}")
                for i, det in enumerate(detections):
                    print(f"    {i+1}. {det['value']}-{det['suit']} ({det['status']}) "
                          f"[{det['value_score']:.2f}/{det['suit_score']:.2f}]")
            else:
                print("\n✗ No hay cartas para capturar")
        
        # C: Calibrar
        elif key == ord('c') or key == ord('C'):
            if not calibration_mode:
                calibration_mode = True
                calibration_points = []
                print("\n→ CALIBRACIÓN ACTIVADA")
            else:
                if len(calibration_points) == 2:
                    lower, upper = calibrate_green_interactive(frame, calibration_points)
                    if lower is not None and upper is not None:
                        LOWER_GREEN = lower
                        UPPER_GREEN = upper
                        print(f"\n✓ Calibración actualizada:")
                        print(f"  LOWER = {list(LOWER_GREEN)}")
                        print(f"  UPPER = {list(UPPER_GREEN)}")
                    else:
                        print("\n✗ Calibración fallida")
                
                calibration_mode = False
                calibration_points = []
                print("→ DETECCIÓN ACTIVADA\n")
        
        # R: Reset
        elif key == ord('r') or key == ord('R'):
            LOWER_GREEN = np.array([35, 40, 40])
            UPPER_GREEN = np.array([85, 255, 255])
            print("\n✓ Calibración reseteada")
        
        # Q/ESC: Salir
        elif key == ord('q') or key == ord('Q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Capturas: {capture_count}")
    if capture_count > 0:
        print(f"Carpeta: {CAPTURE_DIR}/")
        print(f"Log: {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()