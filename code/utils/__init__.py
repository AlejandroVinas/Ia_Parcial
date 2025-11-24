"""
utils/__init__.py
Paquete de utilidades para el sistema de reconocimiento de cartas

Este módulo proporciona funciones auxiliares para:
- Transformaciones geométricas (geometry.py)
- Procesamiento de imágenes (image_processing.py)
- Template matching avanzado (matching.py)

Autor: [Tu Nombre]
Fecha: Noviembre 2025
"""

# Importar funciones de geometría
from .geometry import (
    order_points,
    four_point_transform,
    compute_aspect_ratio,
    is_card_shaped
)

# Importar funciones de procesamiento de imágenes
from .image_processing import (
    load_templates,
    match_best,
    count_pips,
    preprocess_image
)

# Importar funciones de matching
from .matching import (
    template_matching_multi_scale,
    adaptive_threshold_otsu
)

# Definir qué se exporta con "from utils import *"
__all__ = [
    # Geometría
    'order_points',
    'four_point_transform',
    'compute_aspect_ratio',
    'is_card_shaped',
    
    # Procesamiento de imágenes
    'load_templates',
    'match_best',
    'count_pips',
    'preprocess_image',
    
    # Matching
    'template_matching_multi_scale',
    'adaptive_threshold_otsu'
]

# Información del paquete
__version__ = '1.0.0'
__author__ = '[Tu Nombre]'
__description__ = 'Utilidades para reconocimiento de cartas con visión clásica'