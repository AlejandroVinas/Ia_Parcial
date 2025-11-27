# c:\Users\aleja\OneDrive\Escritorio\IA_PARCIAL\v4\code\utils\geometry.py

import numpy as np
import cv2

# Parámetros estándar para una carta de póker (aspect ratio aproximado)
# Nota: La razón de aspecto de una carta de poker es 2.5:3.5, lo que da un ratio de 0.714
CARD_MAX_RATIO = 0.75
CARD_MIN_RATIO = 0.65
WARP_W, WARP_H = 200, 300 # Dimensiones de salida para el Warping

def order_points(pts):
	"""
	Ordena un conjunto de cuatro puntos (esquinas de la carta) de manera consistente:
	(top-left, top-right, bottom-right, bottom-left).
	"""
	rect = np.zeros((4, 2), dtype="float32")

	# Top-Left tiene la suma más pequeña, Bottom-Right la más grande.
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)] 
	rect[2] = pts[np.argmax(s)] 

	# Top-Right tiene la diferencia (y-x) más pequeña, Bottom-Left la más grande.
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)] 
	rect[3] = pts[np.argmax(diff)] 

	return rect

def four_point_transform(image, pts):
	"""
	Aplica una transformación de perspectiva (warping) a la imagen
	basándose en 4 puntos para obtener una vista frontal.
	"""
	# 1. Obtiene los puntos ordenados
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	
	# 2. Define los puntos de destino (la carta normalizada)
	dst = np.array([
		[0, 0],
		[WARP_W - 1, 0],
		[WARP_W - 1, WARP_H - 1],
		[0, WARP_H - 1]], dtype="float32")

	# 3. Calcula y aplica la transformación
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (WARP_W, WARP_H))

	return warped

def compute_aspect_ratio(contour):
	"""
	Calcula la razón de aspecto del rectángulo delimitador de un contorno.
	"""
	# Obtiene el rectángulo delimitador (x, y, ancho, alto)
	_, _, w, h = cv2.boundingRect(contour)
	
	# Calcula la razón de aspecto (ancho/alto)
	# Se asegura de no dividir por cero
	if h > 0:
		aspect_ratio = w / float(h)
	else:
		aspect_ratio = 0.0
	
	# Asegura que sea <= 1 para comparar siempre verticalmente (más pequeño/más grande)
	if aspect_ratio > 1.0:
		aspect_ratio = 1.0 / aspect_ratio
		
	return aspect_ratio, w * h # Retorna ratio y área para validación

def is_card_shaped(aspect_ratio, area_pixels, min_area_ratio):
	"""
	Verifica si un contorno detectado tiene forma y tamaño de carta.
	
	Args:
		aspect_ratio (float): Razón de aspecto (W/H o H/W, siempre <= 1).
		area_pixels (int): Área del contorno en píxeles.
		min_area_ratio (float): Área mínima relativa a la imagen total.
		
	Returns:
		bool: True si es probable que sea una carta.
	"""
	# 1. Validación de la razón de aspecto
	ratio_ok = (aspect_ratio >= CARD_MIN_RATIO and aspect_ratio <= CARD_MAX_RATIO)
	
	# 2. Validación del área (el área es relativa, pero aquí se usa el valor en píxeles)
	# Nota: Es crucial que el 'min_area_ratio' se calcule correctamente en el script principal
	# basado en la resolución de la cámara.
	area_ok = area_pixels > min_area_ratio # Aquí min_area_ratio debería ser el área mínima en PÍXELES
	
	# En el script principal, la validación de área suele usar el área del contorno.
	return ratio_ok and area_ok