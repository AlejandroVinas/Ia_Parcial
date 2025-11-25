#!/usr/bin/env python3
"""
realtime_detection.py
Detección y reconocimiento de cartas en TIEMPO REAL usando webcam externa.

Características:
- Captura desde webcam en tiempo real
- Detección y reconocimiento automático continuo
- Visualización de resultados en pantalla
- Captura de fotos con tecla ESPACIO
- Guardado automático de resultados

Uso:
    python realtime_detection.py --templates ../templates/ --camera 0

Controles:
    ESPACIO: Capturar foto y guardar detección
    C: Calibrar color verde del tapete
    R: Resetear calibración
    Q/ESC: Salir

Autor: [Tu Nombre]
Fecha: Noviembre 2025
"""

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
import pandas as pd


# Importar funciones propias
from utils.geometry import order_points, four_point_transform
from utils.image_processing import load_templates, match_best, count_pips

# ========== PARÁMETROS GLOBALES ==========
# Calibración HSV (ajustables)
LOWER_GREEN = np.array([91, 116, 150])
UPPER_GREEN = np.array([97, 171, 224])

# Configuración de detección
WARP_W, WARP_H = 200, 300
AREA_MIN_RATIO = 0.0008
MIN_VALUE_SCORE = 0.40
MIN_SUIT_SCORE = 0.40

# Configuración de captura
CAPTURE_DIR = "captures"
RESULTS_CSV = "captures/captures_log.csv"

# Estado de calibración
calibration_mode = False
calibration_points = []

# ========== FUNCIONES DE DETECCIÓN ==========

def preprocess_frame(frame, lower_hsv, upper_hsv):
    """Preprocesa frame para segmentación"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_cards = cv2.bitwise_not(mask_green)
    
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def detect_cards_realtime(frame, templates, lower_hsv, upper_hsv, draw=True):
    """
    Detecta y reconoce cartas en un frame de video
    
    Returns:
        detections: Lista de detecciones
        annotated_frame: Frame con anotaciones (si draw=True)
    """
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
        
        if area < AREA_MIN_RATIO * img_area:
            continue
        
        # Aproximación poligonal
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) >= 4:
            if len(approx) > 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                pts = box.astype("float32")
            else:
                pts = approx.reshape(4, 2).astype("float32")
            
            # Transformación de perspectiva
            try:
                warped, M = four_point_transform(frame, pts, WARP_W, WARP_H)
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            except:
                continue
            
            # Extraer ROIs
            roi_val = gray[10:70, 10:60]
            roi_suit = gray[60:120, 10:60]
            
            # Detectar rotación
            if np.mean(roi_val) > 240:
                roi_val = gray[-70:-10, -60:-10]
                roi_suit = gray[-120:-60, -60:-10]
            
            # Matching
            val_name, val_score = match_best(templates.get('values', {}), roi_val)
            suit_name, suit_score = match_best(templates.get('suits', {}), roi_suit)
            
            # Determinar estado
            if val_score >= MIN_VALUE_SCORE and suit_score >= MIN_SUIT_SCORE:
                status = "OK"
                color = (0, 255, 0)  # Verde
            else:
                status = "LOW_CONF"
                color = (0, 165, 255)  # Naranja
            
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
            
            # Dibujar en frame
            if draw:
                # Contorno
                cv2.polylines(annotated, [pts.astype(int)], True, color, 3)
                
                # Texto con resultado
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{val_name or '?'}-{suit_name or '?'}"
                
                # Fondo para texto
                (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (cx - w_text//2 - 5, cy - h_text - 5), 
                             (cx + w_text//2 + 5, cy + 5), (0, 0, 0), -1)
                
                # Texto
                cv2.putText(annotated, text, (cx - w_text//2, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Scores debajo
                score_text = f"{val_score:.2f}/{suit_score:.2f}"
                cv2.putText(annotated, score_text, (cx - w_text//2, cy + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return detections, annotated


def draw_hud(frame, detections, fps, calibrating=False):
    """Dibuja HUD (Heads-Up Display) con información"""
    h, w = frame.shape[:2]
    
    # Panel superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Título
    title = "CALIBRANDO TAPETE VERDE" if calibrating else "DETECTOR DE CARTAS EN TIEMPO REAL"
    cv2.putText(frame, title, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if calibrating else (255, 255, 255), 2)
    
    # Información
    info_text = f"FPS: {fps:.1f} | Cartas: {len(detections)}"
    cv2.putText(frame, info_text, (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Controles en la parte inferior
    controls_y = h - 100
    cv2.rectangle(frame, (0, controls_y), (w, h), (0, 0, 0), -1)
    
    controls = [
        "ESPACIO: Capturar",
        "C: Calibrar",
        "R: Reset",
        "Q/ESC: Salir"
    ]
    
    for i, ctrl in enumerate(controls):
        x_pos = 20 + (i * (w // 4))
        cv2.putText(frame, ctrl, (x_pos, controls_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Lista de detecciones
    if detections and not calibrating:
        y_offset = controls_y - 20
        for i, det in enumerate(detections[-3:]):  # Últimas 3 detecciones
            card_text = f"{i+1}. {det['value']}-{det['suit']} ({det['status']})"
            color = (0, 255, 0) if det['status'] == 'OK' else (0, 165, 255)
            cv2.putText(frame, card_text, (w - 250, y_offset - (i * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def calibrate_green_interactive(frame, points):
    """Calibración interactiva del verde"""
    if len(points) < 2:
        return None, None
    
    # Ordenar puntos
    x1, y1 = min(points[0][0], points[1][0]), min(points[0][1], points[1][1])
    x2, y2 = max(points[0][0], points[1][0]), max(points[0][1], points[1][1])
    
    # Extraer ROI
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None, None
    
    # Convertir a HSV y calcular rango
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


def save_capture(frame, detections, capture_num):
    """Guarda captura con detecciones"""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{capture_num:03d}_{timestamp}.jpg"
    filepath = os.path.join(CAPTURE_DIR, filename)
    
    # Guardar imagen
    cv2.imwrite(filepath, frame)
    
    # Guardar en CSV
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
    
    # Agregar al CSV
    df = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)
    
    return filename


def mouse_callback(event, x, y, flags, param):
    """Callback para calibración con ratón"""
    global calibration_points, calibration_mode
    
    if calibration_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP and len(calibration_points) == 1:
            calibration_points.append((x, y))


# ========== MAIN ==========

def main():
    global calibration_mode, calibration_points, LOWER_GREEN, UPPER_GREEN
    
    parser = argparse.ArgumentParser(
        description="Detección de cartas en tiempo real con webcam"
    )
    parser.add_argument("--templates", required=True, 
                       help="Carpeta con plantillas")
    parser.add_argument("--camera", type=int, default=0, 
                       help="ID de cámara (0=webcam interna, 1=externa)")
    parser.add_argument("--width", type=int, default=1280, 
                       help="Ancho de captura")
    parser.add_argument("--height", type=int, default=720, 
                       help="Alto de captura")
    args = parser.parse_args()
    
    # Cargar plantillas
    print("=" * 60)
    print("CARGANDO PLANTILLAS")
    print("=" * 60)
    templates = load_templates(args.templates)
    
    if not templates['values'] or not templates['suits']:
        print("ERROR: No se cargaron plantillas correctamente")
        return
    
    print(f"✓ Plantillas cargadas: {len(templates['values'])} values, {len(templates['suits'])} suits")
    
    # Inicializar cámara
    print(f"\nAbriendo cámara {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {args.camera}")
        print("Prueba con --camera 1 si tienes una webcam externa")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Cámara abierta: {actual_width}x{actual_height}")
    
    # Crear ventana
    window_name = "Detector de Cartas - Tiempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n" + "=" * 60)
    print("MODO TIEMPO REAL ACTIVO")
    print("=" * 60)
    print("Controles:")
    print("  ESPACIO: Capturar foto")
    print("  C: Calibrar color verde")
    print("  R: Resetear calibración")
    print("  Q/ESC: Salir")
    print("=" * 60 + "\n")
    
    # Variables de control
    capture_count = 0
    fps = 0
    prev_time = time.time()
    
    # Loop principal
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: No se pudo leer frame de la cámara")
            break
        
        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Modo calibración
        if calibration_mode:
            display_frame = frame.copy()
            
            # Dibujar rectángulo de selección
            if len(calibration_points) == 2:
                cv2.rectangle(display_frame, calibration_points[0], calibration_points[1], 
                             (0, 255, 0), 2)
            
            # Instrucciones
            cv2.putText(display_frame, "ARRASTRA para seleccionar area verde", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Presiona C cuando termines", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
        
        # Modo detección normal
        else:
            # Detectar cartas
            detections, annotated_frame = detect_cards_realtime(
                frame, templates, LOWER_GREEN, UPPER_GREEN, draw=True
            )
            
            # Dibujar HUD
            draw_hud(annotated_frame, detections, fps, calibrating=False)
            
            # Mostrar
            cv2.imshow(window_name, annotated_frame)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        # ESPACIO: Capturar
        if key == ord(' ') and not calibration_mode:
            detections, annotated_frame = detect_cards_realtime(
                frame, templates, LOWER_GREEN, UPPER_GREEN, draw=True
            )
            
            if detections:
                capture_count += 1
                filename = save_capture(annotated_frame, detections, capture_count)
                print(f"✓ Captura {capture_count} guardada: {filename}")
                print(f"  Cartas detectadas: {len(detections)}")
                for i, det in enumerate(detections):
                    print(f"    {i+1}. {det['value']}-{det['suit']} ({det['status']})")
            else:
                print("✗ No hay cartas detectadas para capturar")
        
        # C: Calibrar
        elif key == ord('c') or key == ord('C'):
            if not calibration_mode:
                calibration_mode = True
                calibration_points = []
                print("\n→ MODO CALIBRACIÓN activado")
                print("  Arrastra el ratón sobre el tapete verde")
            else:
                # Finalizar calibración
                if len(calibration_points) == 2:
                    lower, upper = calibrate_green_interactive(frame, calibration_points)
                    if lower is not None and upper is not None:
                        LOWER_GREEN = lower
                        UPPER_GREEN = upper
                        print(f"✓ Calibración actualizada:")
                        print(f"  LOWER = {list(LOWER_GREEN)}")
                        print(f"  UPPER = {list(UPPER_GREEN)}")
                    else:
                        print("✗ Calibración fallida, mantiene valores anteriores")
                
                calibration_mode = False
                calibration_points = []
                print("→ MODO DETECCIÓN activado\n")
        
        # R: Reset calibración
        elif key == ord('r') or key == ord('R'):
            LOWER_GREEN = np.array([35, 40, 40])
            UPPER_GREEN = np.array([85, 255, 255])
            print("\n✓ Calibración reseteada a valores por defecto")
        
        # Q/ESC: Salir
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("\nCerrando...")
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Capturas guardadas: {capture_count}")
    if capture_count > 0:
        print(f"Carpeta: {CAPTURE_DIR}/")
        print(f"Log CSV: {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()