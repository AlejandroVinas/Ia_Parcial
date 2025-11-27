# c:\Users\aleja\OneDrive\Escritorio\IA_PARCIAL\v4\code\utils\image_processing.py

import cv2
import numpy as np
import os

TEMPLATE_W, TEMPLATE_H = 200, 300 

def preprocess_image(roi_image):
    """
    Preprocesa la Región de Interés (ROI) para el Template Matching.
    Aplica Umbral de Otsu.
    """
    blur = cv2.GaussianBlur(roi_image, (3, 3), 0)
    # THRESH_OTSU calcula el umbral óptimo automáticamente.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def load_templates(templates_dir):
    """
    Carga, redimensiona y preprocesa todas las plantillas (valores y palos) 
    desde el directorio especificado.
    """
    templates = {
        'values': {},
        'suits': {}
    }

    # Cargar valores
    values_dir = os.path.join(templates_dir, 'values')
    for filename in os.listdir(values_dir):
        if filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(values_dir, filename)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                template = cv2.resize(template, (TEMPLATE_W, TEMPLATE_H))
                # Extracción y preprocesamiento de la ROI de valor
                roi_template = template[10:70, 10:60]
                templates['values'][name] = preprocess_image(roi_template)
    
    # Cargar palos
    suits_dir = os.path.join(templates_dir, 'suits')
    for filename in os.listdir(suits_dir):
        if filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(suits_dir, filename)
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                template = cv2.resize(template, (TEMPLATE_W, TEMPLATE_H))
                # Extracción y preprocesamiento de la ROI de palo
                roi_template = template[60:120, 10:60]
                templates['suits'][name] = preprocess_image(roi_template)

    return templates


def match_best(templates, roi_image):
    """
    Compara la ROI de la carta con todas las plantillas usando Template Matching.
    """
    best_score = -1.0
    best_name = None
    
    # Preprocesar la ROI del frame antes de comparar
    processed_roi = preprocess_image(roi_image) 

    for name, template in templates.items():
        if template.shape != processed_roi.shape:
            continue

        result = cv2.matchTemplate(processed_roi, template, cv2.TM_CCOEFF_NORMED)
        
        _, max_score, _, _ = cv2.minMaxLoc(result)
        
        if max_score > best_score:
            best_score = max_score
            best_name = name
            
    return best_name, best_score


def count_pips(warped_card):
    """Función placeholder. El reconocimiento se hace por Template Matching."""
    return 0

if __name__ == '__main__':
    print("Módulo image_processing cargado correctamente.")