import cv2
import numpy as np
import imutils
from utils import four_point_transform, extract_corner, detect_symbols

def find_cards(frame):
    # Convertir a gris y aplicar blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Encontrar contornos
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cards = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2500:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            warped = four_point_transform(frame, pts)
            cards.append({'pts': pts, 'warped': warped})

    return cards

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print('No se puede abrir la cámara (índice 1)')
        return

    print('Presiona q para salir')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer la cámara")
            break

        frame = imutils.resize(frame, width=900)
        cards = find_cards(frame)

        for c in cards:
            warped = c['warped']
            corner = extract_corner(warped)
            num_symbols, suit = detect_symbols(corner)

            label = f"Símbolos: {num_symbols}, Palo: {suit}"
            cv2.polylines(frame, [c['pts']], True, (0, 255, 0), 2)

            # Asegurarse que las coordenadas sean enteros
            x, y = c['pts'][0]
            cv2.putText(frame, label, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Detector de cartas (formas)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
