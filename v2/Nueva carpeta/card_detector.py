import cv2
import numpy as np
import imutils
from utils import four_point_transform, extract_corner, detect_symbols

def create_green_mask(frame):
    """Crea una máscara para aislar el tapete verde y facilitar detección de cartas (MEJORADO)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- Rangos HSV ajustados para una mayor robustez ---
    # Se amplía el rango de Tono (H) y se relaja el de Saturación (S) y Valor (V)
    
    # Rango de verde principal (35-85)
    lower_green1 = np.array([25, 30, 30])  # Tono más bajo, Saturación y Valor más bajos
    upper_green1 = np.array([90, 255, 255])
    
    mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Invertir: queremos las cartas (no verde)
    mask_cards = cv2.bitwise_not(mask_green)
    
    # Limpiar ruido: se aumenta el tamaño del kernel para una limpieza más agresiva.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) # Kernel aumentado a (7, 7)
    mask_cards = cv2.morphologyEx(mask_cards, cv2.MORPH_CLOSE, kernel, iterations=2) # Más iteraciones de CLOSING
    mask_cards = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask_cards

def find_cards(frame):
    """Detecta cartas sobre tapete verde usando segmentación por color (MEJORADO)"""
    
    mask = create_green_mask(frame)
    
    # Aplicar la máscara al frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Umbral binario: Reducir el umbral a 10 para capturar cartas más oscuras o con sombras.
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para conectar componentes de la carta
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) # Kernel más grande
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=3) # Más iteraciones de CLOSING
    thresh = cv2.dilate(thresh, kernel_large, iterations=2) # Más dilatación para unir bien los bordes

    # Encontrar contornos
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cards = []
    for c in cnts:
        area = cv2.contourArea(c)
        
        # Filtrar por área (se ajustan los límites para ser más inclusivos)
        if area < 6000 or area > 300000: # Se reduce el mínimo (de 8000 a 6000) y se aumenta el máximo
            continue

        # Aproximar el contorno a un polígono
        peri = cv2.arcLength(c, True)
        # Se reduce el factor de aproximación (0.015 a 0.01) para una aproximación más precisa
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # Verificar que sea un cuadrilátero o un pentágono (a veces un reflejo genera 5 vértices)
        if 4 <= len(approx) <= 5:
            # Verificar relación de aspecto de una carta estándar
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Rangos de relación de aspecto más amplios: 0.4 a 2.5
            if 0.4 < aspect_ratio < 2.5: 
                # Verificar que tenga suficiente "blancura"
                mask_roi = mask[y:y+h, x:x+w]
                if mask_roi.size > 0:
                    # Se reduce el umbral de blanco al 50% (de 60% a 50%)
                    white_percentage = np.sum(mask_roi > 0) / mask_roi.size
                    
                    if white_percentage > 0.5: # Al menos 50% blanco
                        pts = approx.reshape(len(approx), 2)
                        
                        # Si es un pentágono, se intenta reducir a un cuadrilátero
                        if len(pts) == 5:
                            # Se usa el bounding box de 4 puntos en lugar de los 5 aproximados para la transformación
                            rect_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype="float32")
                            # La función four_point_transform espera 4 puntos, debemos asegurarnos de pasar 4 puntos.
                            # Aquí se podría usar un filtro para elegir los 4 vértices más prominentes.
                            # Para simplificar y mantener la robustez, usamos la aproximación *original* (4 puntos)
                            # y si se pasa a 5, se salta. **Volvemos a exigir estrictamente 4 vértices para la perspectiva.**
                            continue 
                        
                        warped = four_point_transform(frame, pts)
                        cards.append({
                            'pts': pts, 
                            'warped': warped,
                            'area': area,
                            'aspect': aspect_ratio
                        })

    return cards, mask

def main():
    # Intentar abrir la cámara externa (índice 1)
    cap = cv2.VideoCapture(1)
    
    # Si no funciona, intentar cámara integrada
    if not cap.isOpened():
        print('No se puede abrir la cámara externa (índice 1), intentando con índice 0...')
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print('No se puede abrir ninguna cámara')
        return

    # Configurar resolución (Se mantiene)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Ajustar exposición para mejor contraste (Se mantiene)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    print('===========================================')
    print('Detector de Cartas - Tapete Verde (MEJORADO)')
    print('===========================================')
    print('Optimizado para tapete verde de póker')
    print('')
    print('Instrucciones:')
    print('  - Coloca las cartas sobre el tapete verde')
    print('  - Asegúrate de buena iluminación uniforme')
    print('  - Evita sombras sobre las cartas')
    print('')
    print('Controles:')
    print('  - "q": Salir')
    print('  - "d": Mostrar debug (máscaras y procesamiento)')
    print('  - "m": Mostrar solo máscara de detección')
    print('===========================================\n')

    show_debug = False
    show_mask_only = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer de la cámara")
            break

        # Redimensionar para mejor rendimiento (Se mantiene)
        frame = imutils.resize(frame, width=1100)
        display_frame = frame.copy()
        
        # Detectar cartas
        cards, detection_mask = find_cards(frame)

        # Procesar cada carta detectada
        for idx, card_data in enumerate(cards):
            warped = card_data['warped']
            pts = card_data['pts']
            
            # Extraer esquina superior izquierda
            corner = extract_corner(warped)
            
            # Detectar valor y palo
            rank, suit = detect_symbols(corner)

            # Preparar etiqueta con información adicional
            label = f"{rank} de {suit}"
            
            # Color del contorno según confianza
            # Se añade 'Figura' a la condición de baja confianza
            if 'Desconocido' in rank or 'Desconocido' in suit or 'Figura' in rank or 'Número' in rank:
                color = (0, 165, 255)  # Naranja para baja confianza o clasificaciones genéricas
            else:
                color = (0, 255, 0)  # Verde para buena detección
            
            # Dibujar contorno de la carta
            cv2.polylines(display_frame, [pts], True, color, 3)

            # Colocar etiqueta con fondo para mejor legibilidad (Se mantiene)
            x, y = pts[0]
            
            # Asegurar que las coordenadas son enteras para cv2.rectangle y cv2.putText
            x_int, y_int = int(x), int(y)

            # Fondo semi-transparente para el texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_frame, 
                         (x_int, y_int - 35),
                         (x_int + text_size[0] + 10, y_int - 5),
                         (0, 0, 0), -1)
            
            cv2.putText(display_frame, label, (x_int + 5, y_int - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Círculo para indicar esquina superior izquierda
            cv2.circle(display_frame, (x_int, y_int), 6, (255, 0, 0), -1)
            
            # Modo debug: mostrar cartas procesadas
            if show_debug and idx < 4:  # Limitar a 4 cartas
                # Obtener la imagen de umbral para el debug
                _, _, thresh_corner = separate_rank_and_suit(corner)
                
                cv2.imshow(f'Debug {idx+1} - Carta', imutils.resize(warped, width=200))
                cv2.imshow(f'Debug {idx+1} - Esquina THRESH', thresh_corner) # Mostrar umbral de esquina
                cv2.imshow(f'Debug {idx+1} - Esquina RGB', corner) # Mostrar la esquina original

        # Mostrar solo máscara si está activado (Se mantiene)
        if show_mask_only:
            mask_colored = cv2.cvtColor(detection_mask, cv2.COLOR_GRAY2BGR)
            # Combinar la imagen original y la máscara
            display_frame = cv2.addWeighted(frame, 0.5, mask_colored, 0.5, 0)

        # Información en pantalla (Se mantiene)
        info_text = f"Cartas: {len(cards)} | q:Salir | d:Debug | m:Mascara"
        cv2.rectangle(display_frame, (5, 5), (490, 35), (0, 0, 0), -1)
        cv2.putText(display_frame, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Indicador de modo (Se mantiene)
        if show_debug:
            cv2.putText(display_frame, "DEBUG ON", (display_frame.shape[1] - 120, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if show_mask_only:
            cv2.putText(display_frame, "MASK VIEW", (display_frame.shape[1] - 130, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Mostrar frame principal
        cv2.imshow('Detector de Cartas - Tapete Verde', display_frame)

        # Manejar teclas (Se mantiene)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Modo debug: {'ACTIVADO' if show_debug else 'DESACTIVADO'}")
            if not show_debug:
                # Cerrar ventanas de debug
                for i in range(4):
                    try:
                        cv2.destroyWindow(f'Debug {i+1} - Carta')
                        cv2.destroyWindow(f'Debug {i+1} - Esquina RGB')
                        cv2.destroyWindow(f'Debug {i+1} - Esquina THRESH')
                    except:
                        pass
        elif key == ord('m'):
            show_mask_only = not show_mask_only
            print(f"Vista de máscara: {'ACTIVADA' if show_mask_only else 'DESACTIVADA'}")

    # Liberar recursos (Se mantiene)
    cap.release()
    cv2.destroyAllWindows()
    print('\nPrograma finalizado')

if __name__ == '__main__':
    main()