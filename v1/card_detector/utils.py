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

def extract_corner(warped_card, corner_size=(100, 100)):
    h, w = warped_card.shape[:2]
    ch, cw = corner_size
    corner = warped_card[0:ch, 0:cw]
    return corner

def classify_shape(contour):
    # Clasifica el tipo de símbolo según su forma geométrica
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    if area < 50:
        return None

    if len(approx) > 6:
        return 'Corazón/Diamante'
    elif 4 <= len(approx) <= 6:
        return 'Trébol/Pica'
    else:
        return 'Desconocido'

def detect_symbols(corner):
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    for c in contours:
        shape = classify_shape(c)
        if shape:
            symbols.append(shape)

    # Contar símbolos para estimar valor (aproximado)
    num_symbols = len(symbols)
    suit_type = symbols[0] if symbols else 'Desconocido'

    return num_symbols, suit_type
