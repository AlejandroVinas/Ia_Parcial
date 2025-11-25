#!/usr/bin/env python3
"""
detect_and_recognize.py
Pipeline completo para detección y reconocimiento clásico de cartas sobre tapete verde.
Salida: CSV con detecciones y carpeta debug/ con imágenes intermedias.

Uso:
    python detect_and_recognize.py --input test_images/ --templates templates/ --out results.csv

Requisitos:
    pip install opencv-python numpy pandas

Autor: [Tu nombre]
Fecha: 2025
Examen Parcial - Visión Artificial
"""
import os
import cv2
import numpy as np
import argparse
import glob
import pandas as pd

# ========== PARÁMETROS CONFIGURABLES ==========
# Calibrar estos valores usando calibrate_hsv.py
LOWER_GREEN = np.array([35, 40, 40])   # Límite inferior HSV para verde
UPPER_GREEN = np.array([85, 255, 255]) # Límite superior HSV para verde

# Dimensiones de warping normalizadas
WARP_W, WARP_H = 200, 300

# Umbral de área mínima (relativo al área total de imagen)
AREA_MIN_RATIO = 0.0008

# Umbrales de confianza para matching
MIN_VALUE_SCORE = 0.40  # Score mínimo para considerar válido el valor
MIN_SUIT_SCORE = 0.40   # Score mínimo para considerar válido el palo

# Directorio de salida para debug
DEBUG_DIR = "../debug"

# Columnas del CSV de resultados
CSV_COLUMNS = ['filename', 'card_id', 'value', 'value_score', 'suit', 
               'suit_score', 'area', 'x_min', 'y_min', 'x_max', 'y_max', 'status']


# ========== FUNCIONES DE GEOMETRÍA ==========
def order_points(pts):
    """
    Ordena 4 puntos en el orden: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array numpy de 4 puntos (x,y)
    
    Returns:
        Array ordenado de 4 puntos
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left tendrá la suma más pequeña
    # Bottom-right tendrá la suma más grande
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right tendrá la diferencia más pequeña
    # Bottom-left tendrá la diferencia más grande
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts, W=WARP_W, H=WARP_H):
    """
    Realiza transformación de perspectiva para enderezar carta
    
    Args:
        image: Imagen original
        pts: 4 puntos de la carta detectada
        W, H: Dimensiones de salida
    
    Returns:
        warped: Imagen transformada
        M: Matriz de transformación
    """
    rect = order_points(pts)
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped, M


# ========== FUNCIONES DE CARGA DE PLANTILLAS ==========
def load_templates(folder):
    """
    Carga plantillas de valores y palos desde carpeta estructurada
    
    Estructura esperada:
        templates/
            values/
                2.png, 3.png, ..., 10.png, J.png, Q.png, K.png, A.png
            suits/
                hearts.png, diamonds.png, clubs.png, spades.png
    
    Args:
        folder: Ruta a carpeta templates/
    
    Returns:
        Dict con {'values': {nombre: imagen}, 'suits': {nombre: imagen}}
    """
    templates = {'values': {}, 'suits': {}}
    
    for kind in ['values', 'suits']:
        path = os.path.join(folder, kind)
        if not os.path.isdir(path):
            print(f"ADVERTENCIA: No se encontró carpeta {path}")
            continue
        
        for filepath in glob.glob(os.path.join(path, "*.png")):
            name = os.path.splitext(os.path.basename(filepath))[0]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"ADVERTENCIA: No se pudo cargar {filepath}")
                continue
            
            # Binarizar plantilla con umbral fijo
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            templates[kind][name] = img_bin
            
        print(f"Cargadas {len(templates[kind])} plantillas de {kind}")
    
    return templates


# ========== FUNCIONES DE MATCHING ==========
def match_best(template_dict, roi_gray):
    """
    Encuentra la mejor coincidencia entre un ROI y un diccionario de plantillas
    
    Proceso:
    1. Binariza el ROI con umbral Otsu
    2. Para cada plantilla:
       - Ajusta tamaño si es necesario
       - Realiza template matching con correlación normalizada
    3. Retorna la plantilla con mejor score
    
    Args:
        template_dict: Dict {nombre: plantilla_binarizada}
        roi_gray: ROI en escala de grises
    
    Returns:
        (nombre_mejor, score_mejor): Tupla con resultado
    """
    # Binarizar ROI con método adaptativo Otsu
    _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    best_name, best_score = None, -1.0
    
    for name, tpl in template_dict.items():
        tpl_h, tpl_w = tpl.shape
        r_h, r_w = roi_bin.shape
        
        # Redimensionar plantilla si es más grande que el ROI
        if tpl_h > r_h or tpl_w > r_w:
            scale_h = r_h / tpl_h
            scale_w = r_w / tpl_w
            scale = min(scale_h, scale_w) * 0.9  # 90% para margen
            newsize = (max(1, int(tpl_w * scale)), max(1, int(tpl_h * scale)))
            tpl_resized = cv2.resize(tpl, newsize, interpolation=cv2.INTER_AREA)
        else:
            tpl_resized = tpl
        
        # Template matching con correlación normalizada
        try:
            res = cv2.matchTemplate(roi_bin, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, _ = cv2.minMaxLoc(res)
            
            if maxval > best_score:
                best_score = maxval
                best_name = name
        except Exception as e:
            # ROI o template demasiado pequeño
            continue
    
    return best_name, float(best_score)


def count_pips(warp_gray):
    """
    Cuenta pips (símbolos) en el cuerpo de la carta para validación
    
    Útil para validar cartas numéricas (2-10)
    
    Args:
        warp_gray: Carta enderezada en escala de grises
    
    Returns:
        count: Número de componentes detectados
    """
    h, w = warp_gray.shape
    
    # Extraer región del cuerpo (excluir esquinas)
    body = warp_gray[int(h*0.20):int(h*0.85), int(w*0.15):int(w*0.85)]
    
    # Binarizar
    _, th = cv2.threshold(body, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apertura morfológica para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Componentes conectados (invertir imagen para detectar símbolos oscuros)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - clean, connectivity=8)
    
    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Heurística: pips tienen tamaño característico
        # Ajustar según resolución y tipo de cartas
        if 20 < area < 3000 and 5 < w_box < 120 and 5 < h_box < 120:
            count += 1
    
    return count


# ========== PIPELINE PRINCIPAL ==========
def process_image(path, templates, out_debug=True):
    """
    Procesa una imagen para detectar y reconocer cartas
    
    Pipeline:
    1. Carga imagen
    2. Segmentación por color (HSV) para separar cartas de fondo verde
    3. Detección de contornos
    4. Para cada contorno válido:
       - Aproximación poligonal
       - Transformación de perspectiva
       - Extracción de ROIs de esquina
       - Matching con plantillas
       - Validación
    
    Args:
        path: Ruta a imagen
        templates: Dict de plantillas cargadas
        out_debug: Si True, guarda imágenes de debug
    
    Returns:
        Lista de detecciones: [{'card_id', 'value', 'suit', ...}, ...]
    """
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: No se pudo cargar {path}")
        return []
    
    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img
    
    # ===== PASO 1: SEGMENTACIÓN =====
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Máscara de verde (tapete)
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # Invertir para obtener cartas (no-verde)
    mask_cards = cv2.bitwise_not(mask_green)
    
    # Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ===== PASO 2: DETECCIÓN DE CONTORNOS =====
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    img_debug = img.copy()
    card_idx = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtrar por área mínima
        if area < AREA_MIN_RATIO * img_area:
            continue
        
        # ===== PASO 3: APROXIMACIÓN POLIGONAL =====
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Necesitamos al menos 4 vértices (cuadrilátero)
        if len(approx) >= 4:
            if len(approx) > 4:
                # Usar minAreaRect si hay más de 4 vértices
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                pts = box.astype("float32")
            else:
                pts = approx.reshape(4, 2).astype("float32")
            
            # ===== PASO 4: TRANSFORMACIÓN DE PERSPECTIVA =====
            warped, M = four_point_transform(img, pts)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            # ===== PASO 5: EXTRACCIÓN DE ROIs =====
            # ROI para valor (esquina superior)
            roi_val = gray[10:70, 10:60]
            # ROI para palo (debajo del valor)
            roi_suit = gray[60:120, 10:60]
            
            # Verificar si la esquina está vacía (carta rotada 180°)
            mean_brightness = np.mean(roi_val)
            if mean_brightness > 240:  # Casi blanco = esquina vacía
                # Probar esquina opuesta (rotación 180°)
                roi_val = gray[-70:-10, -60:-10]
                roi_suit = gray[-120:-60, -60:-10]
            
            # ===== PASO 6: MATCHING =====
            val_name, val_score = match_best(templates.get('values', {}), roi_val)
            suit_name, suit_score = match_best(templates.get('suits', {}), roi_suit)
            
            # ===== PASO 7: VALIDACIÓN =====
            # Contar pips para validación adicional
            pips = count_pips(gray)
            
            status = "ok"
            
            # Validación básica por scores
            if val_score < MIN_VALUE_SCORE or suit_score < MIN_SUIT_SCORE:
                status = "low_confidence"
            else:
                # Validación cruzada con conteo de pips (solo cartas numéricas)
                try:
                    if val_name and val_name.isdigit():
                        numeric = int(val_name)
                        if pips > 0 and abs(pips - numeric) > 2:
                            status = "pip_mismatch"
                except:
                    pass
            
            # ===== PASO 8: BOUNDING BOX Y ANOTACIÓN =====
            xs, ys = pts[:, 0], pts[:, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            
            # Dibujar en imagen de debug
            cv2.polylines(img_debug, [pts.astype(int)], True, (0, 255, 0), 2)
            cx, cy = int(xs.mean()), int(ys.mean())
            
            text = f"{val_name or '?'}{suit_name or '?'} ({val_score:.2f},{suit_score:.2f})"
            cv2.putText(img_debug, text, (cx-60, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # ===== PASO 9: GUARDAR DETECCIÓN =====
            detections.append({
                'card_id': card_idx,
                'value': val_name if val_name else '',
                'value_score': val_score,
                'suit': suit_name if suit_name else '',
                'suit_score': suit_score,
                'area': area,
                'bbox': (x_min, y_min, x_max, y_max),
                'status': status
            })
            card_idx += 1
    
    # ===== SALIDA DE DEBUG =====
    if out_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base}_mask.png"), mask)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{base}_debug.png"), img_debug)
    
    return detections


# ========== INTERFAZ DE LÍNEA DE COMANDOS ==========
def main():
    parser = argparse.ArgumentParser(
        description="Sistema de reconocimiento de cartas mediante visión clásica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    # Procesar una imagen
    python detect_and_recognize.py --input test.jpg --templates templates/ --out results.csv
    
    # Procesar carpeta completa
    python detect_and_recognize.py --input test_images/ --templates templates/ --out results.csv
        """
    )
    parser.add_argument("--input", required=True, 
                       help="Imagen individual o carpeta con imágenes")
    parser.add_argument("--templates", required=True, 
                       help="Carpeta templates/ con subcarpetas values/ y suits/")
    parser.add_argument("--out", default="results.csv", 
                       help="Archivo CSV de salida (default: results.csv)")
    parser.add_argument("--no-debug", action="store_true",
                       help="No guardar imágenes de debug")
    args = parser.parse_args()
    
    # Cargar plantillas
    print("=" * 60)
    print("CARGANDO PLANTILLAS")
    print("=" * 60)
    templates = load_templates(args.templates)
    
    if not templates['values'] or not templates['suits']:
        print("ERROR: No se cargaron plantillas. Verifica la estructura de carpetas.")
        return
    
    # Obtener lista de imágenes a procesar
    paths = []
    if os.path.isdir(args.input):
        exts = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
        for ext in exts:
            paths.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        paths = [args.input]
    
    if not paths:
        print("ERROR: No se encontraron imágenes para procesar")
        return
    
    print(f"\nEncontradas {len(paths)} imágenes para procesar")
    
    # Procesar cada imagen
    print("\n" + "=" * 60)
    print("PROCESANDO IMÁGENES")
    print("=" * 60)
    
    rows = []
    for idx, path in enumerate(paths, 1):
        print(f"\n[{idx}/{len(paths)}] Procesando: {os.path.basename(path)}")
        dets = process_image(path, templates, out_debug=not args.no_debug)
        print(f"  → Detectadas {len(dets)} cartas")
        
        for d in dets:
            x_min, y_min, x_max, y_max = d['bbox']
            rows.append({
                'filename': os.path.basename(path),
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
    
    # Guardar resultados
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(args.out, index=False)
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes procesadas: {len(paths)}")
    print(f"Cartas detectadas: {len(rows)}")
    print(f"Resultados guardados en: {args.out}")
    if not args.no_debug:
        print(f"Imágenes de debug en: {DEBUG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()