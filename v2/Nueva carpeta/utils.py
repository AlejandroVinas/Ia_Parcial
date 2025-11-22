import cv2
import numpy as np

CARD_WIDTH = 200
CARD_HEIGHT = 300

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, w=CARD_WIDTH, h=CARD_HEIGHT):
    rect = order_points(pts)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def extract_corner(warped_card, corner_size=(90, 130)):
    """Extrae la esquina superior izquierda donde están el número y el palo (Ajuste de margen)"""
    h, w = warped_card.shape[:2]
    ch, cw = corner_size
    # Reducir el margen inicial (8px) a 5px para capturar más del símbolo/valor si están cerca del borde
    # Se aumenta el tamaño de la región a 100x140 para capturar mejor los símbolos grandes
    corner = warped_card[5:min(ch + 10, h), 5:min(cw + 10, w)]
    return corner

def classify_suit_shape(contour): # Se eliminó 'moments' como argumento ya que no se usa en el cuerpo actual, pero se recalcula
    """Clasifica el palo según características geométricas avanzadas (MEJORADO)"""
    area = cv2.contourArea(contour)
    if area < 100: # Se mantiene el umbral de área
        return None
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    
    moments = cv2.moments(contour)
    
    # Calcular circularidad
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Calcular ratio de aspecto del bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Calcular solidez (Solidarity)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Aproximación poligonal
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # --- Criterios de Clasificación Refinados ---
    
    # CORAZONES: Forma redonda/ovalada, circularidad moderada, solidez alta
    if 0.5 < circularity < 0.9 and 0.8 < aspect_ratio < 1.3 and solidity > 0.8:
        # Los corazones pueden tener una solidez ligeramente menor a los tréboles/diamantes por la muesca superior.
        return 'Corazones'
    
    # DIAMANTES: 4 vértices, elongado (vertical u horizontal), solidez alta, baja circularidad
    if 3 <= vertices <= 6 and 0.4 < aspect_ratio < 1.7 and circularity < 0.55 and solidity > 0.8:
        return 'Diamantes'
    
    # TRÉBOLES: Tres 'círculos' conectados, circularidad y solidez moderadas a altas.
    if 0.5 < circularity < 0.8 and 0.8 < aspect_ratio < 1.2 and solidity > 0.85:
        # Alta solidez y circularidad cercana a 0.7-0.8 (más que picas/diamantes)
        return 'Treboles'
    
    # PICAS: Forma de corazón invertido con punta, elongado verticalmente, baja circularidad, solidez moderada.
    if circularity < 0.6 and 0.7 < aspect_ratio < 1.2 and solidity > 0.75 and h > w:
        return 'Picas'
    
    return None

def separate_rank_and_suit(corner):
    """Separa el número/figura del símbolo del palo en la esquina (MEJORADO)"""
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    
    # Aumentar el kernel de GaussianBlur para un mejor suavizado
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Ajustar parámetros de umbral adaptativo
    # 1. Block Size (17): Se mantiene para un umbral local.
    # 2. C (12): Se reduce a 8 para ser menos restrictivo y capturar mejor las figuras complejas y el palo.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 17, 8)
    
    # Operaciones morfológicas para limpiar ruido y rellenar agujeros en símbolos
    kernel_open = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = np.ones((4, 4), np.uint8) # Aumentar kernel para cerrar mejor figuras y palos
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, thresh
    
    # Filtrar contornos por área y reordenar
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        # Reducir un poco el área mínima (de 150 a 100) para no perder números/palos pequeños o rotados.
        if area > 100:
            valid_contours.append(c)
    
    if len(valid_contours) == 0:
        return None, None, thresh
    
    # Ordenar contornos por posición vertical (y)
    bboxes = []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append({'contour': c, 'x': x, 'y': y, 'w': w, 'h': h, 'area': cv2.contourArea(c)})
    
    bboxes.sort(key=lambda b: b['y'])
    
    # El número/figura suele estar arriba, el palo abajo
    rank_contour = None
    suit_contour = None
    
    # Usar los dos contornos más grandes y verticalmente separados.
    if len(bboxes) >= 2:
        # Asumir que el primer contorno (arriba) es el rank y el segundo es el suit
        rank_contour = bboxes[0]['contour']
        suit_contour = bboxes[1]['contour']
    elif len(bboxes) == 1:
        # Si solo hay uno, intentar determinar si es rank (más alto) o suit (más cuadrado/ancho)
        b = bboxes[0]
        aspect = b['h'] / b['w'] if b['w'] > 0 else 0
        if b['h'] > 30 and aspect > 1.2: # Rank es típicamente más alto que ancho
            rank_contour = b['contour']
        else:
            suit_contour = b['contour']
    
    return rank_contour, suit_contour, thresh

def detect_rank(rank_contour, thresh_image):
    """Detecta el valor de la carta (A, 2-10, J, Q, K) (MEJORADO)"""
    if rank_contour is None:
        return 'Desconocido'
    
    x, y, w, h = cv2.boundingRect(rank_contour)
    
    # Extraer la región del número/figura
    rank_roi = thresh_image[y:y+h, x:x+w]
    
    if rank_roi.size == 0:
        return 'Desconocido'
    
    # Calcular densidad de píxeles blancos (solidez de la imagen de umbral)
    density = np.sum(rank_roi == 255) / rank_roi.size
    
    # Calcular relación de aspecto
    aspect_ratio = h / w if w > 0 else 0
    
    # Analizar la complejidad del contorno
    perimeter = cv2.arcLength(rank_contour, True)
    area = cv2.contourArea(rank_contour)
    complexity = perimeter ** 2 / area if area > 0 else 0
    
    # Contar componentes conectados (útil para 8, 0, 4, 6, 9)
    num_labels, labels = cv2.connectedComponents(rank_roi)
    
    # --- Criterios de Clasificación Refinados ---
    
    # AS (A): Forma alta, a menudo con un "pico" o línea vertical fuerte.
    if aspect_ratio > 1.8 and complexity < 28 and density < 0.4:
        return 'A'
    
    # FIGURAS (J, Q, K): Alta complejidad y alta densidad. Son diseños grandes.
    if density > 0.45 or complexity > 40: # Aumento del umbral de complejidad
        return 'Figura (J/Q/K)'
    
    # 8: Dos agujeros, lo que resulta en num_labels > 3 (fondo + 8 + 2 agujeros) o alta densidad.
    if num_labels >= 4 and aspect_ratio > 1.2 and density > 0.4:
        return '8'

    # 4: Típicamente un contorno con una abertura o un hueco, aspect ratio moderado.
    if aspect_ratio < 1.4 and complexity > 25:
        # Se puede distinguir por el número de vértices en una aproximación (4 es complejo)
        return '4'
    
    # 10: Combina la forma de '1' y '0'
    if aspect_ratio < 1.2 and num_labels >= 4:
        return '10'

    # 6 y 9: Un agujero, lo que resulta en num_labels = 3
    if num_labels == 3 and aspect_ratio > 1.2:
        return '6/9'

    # 2, 3, 5, 7
    if aspect_ratio > 1.5:
        if num_labels == 2:
            return '7/1'
        elif num_labels == 3:
            return '2/5'
        elif num_labels == 4:
            return '3'

    return 'Número' # Clasificación genérica para el resto de números.

def detect_symbols(corner):
    """Función principal para detectar el valor y palo de la carta"""
    rank_contour, suit_contour, thresh = separate_rank_and_suit(corner)
    
    # Detectar el valor (número o figura)
    rank = detect_rank(rank_contour, thresh)
    
    # Detectar el palo
    suit = 'Desconocido'
    if suit_contour is not None:
        suit_result = classify_suit_shape(suit_contour) # Ahora solo necesita el contorno
        if suit_result:
            suit = suit_result
    
    # Clasificación de respaldo más simple si la clasificación avanzada falla
    if suit == 'Desconocido' and suit_contour is not None:
        area = cv2.contourArea(suit_contour)
        x, y, w, h = cv2.boundingRect(suit_contour)
        aspect = h / w if w > 0 else 1
        
        # Criterios básicos de respaldo
        if aspect < 0.8:
            suit = 'Diamantes/Treboles' # Más ancho que alto
        elif aspect > 1.3:
            suit = 'Picas/Corazones' # Más alto que ancho
        else:
            suit = 'Corazones/Treboles (Cuadrado)'
    
    return rank, suit