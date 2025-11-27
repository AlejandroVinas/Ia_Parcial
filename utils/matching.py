# c:\Users\aleja\OneDrive\Escritorio\IA_PARCIAL\v4\code\utils\matching.py

import cv2
import numpy as np
import imutils

def adaptive_threshold_otsu(image):
    """
    Aplica una umbralización que combina el método de Otsu (global y robusto)
    o un umbral adaptativo (local), dependiendo de la robustez deseada.

    Para el reconocimiento de cartas normalizadas, un simple umbral de Otsu 
    suele ser muy efectivo para binarizar los símbolos de valor y palo.
    
    Args:
        image (numpy.ndarray): Imagen en escala de grises.
        
    Returns:
        numpy.ndarray: Imagen binarizada.
    """
    # Usar un umbral de Otsu global para limpiar los símbolos blancos/negros
    # El umbral se calcula automáticamente, muy útil para variaciones de iluminación.
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir si es necesario (el texto de las cartas suele ser negro sobre blanco)
    # Por lo general, los símbolos son negros (0) y el fondo blanco (255)
    
    return thresh


def template_matching_multi_scale(image, template, threshold=0.7):
    """
    Realiza Template Matching a través de múltiples escalas.

    Esto es útil si la detección inicial de la carta falla, y queremos buscar 
    el patrón dentro de una región más grande de la imagen a diferentes tamaños,
    lo que ayuda con las variaciones de perspectiva y tamaño (opcional +1 pt).

    Args:
        image (numpy.ndarray): Imagen fuente (región de interés o imagen completa).
        template (numpy.ndarray): Plantilla a buscar.
        threshold (float): Score mínimo de correlación para ser considerado un match.

    Returns:
        tuple: (max_score, max_loc, max_scale) del mejor match.
    """
    found = None
    max_scale = 1.0
    
    # Convertir a escala de grises si no lo está
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Definir el rango de escalas (e.g., de 100% a 50%)
    for scale in np.linspace(1.0, 0.5, 20)[::-1]:
        # Redimensionar la imagen para hacer coincidir el template
        resized = imutils.resize(gray_image, width=int(gray_image.shape[1] * scale))
        
        # El template debe ser más pequeño que la imagen redimensionada
        if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break

        # Realizar template matching
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)

        # Si encontramos una mejor coincidencia, guardarla
        if found is None or max_val > found[0]:
            found = (max_val, max_loc, scale)

    if found and found[0] >= threshold:
        return found
    else:
        return None

if __name__ == '__main__':
    print("Módulo matching.py cargado correctamente.")