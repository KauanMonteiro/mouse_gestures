import cv2
import mediapipe as mp
import pyautogui

# Inicialização do MediaPipe e do pyautogui
hands = mp.solutions.hands
Hands = hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Captura de vídeo da webcam
video = cv2.VideoCapture(0)

# Tamanho da tela
screen_width, screen_height = pyautogui.size()

# Limiar para considerar o polegar abaixado
click_threshold = 0.5  # Ajuste esse valor conforme necessário

while True:
    success, img = video.read()
    img = cv2.flip(img, 1)  # Inverter imagem horizontalmente
    if not success:
        print("Failed to grab frame")
        break
    
    if img is not None:
        # Conversão da imagem para RGB
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hands.process(frameRGB)
        handPoints = results.multi_hand_landmarks
        h, w, _ = img.shape

        if handPoints:
            for points in handPoints:
                # Desenhar os pontos e conexões na imagem
                mpDraw.draw_landmarks(img, points, hands.HAND_CONNECTIONS)

                # Posições dos pontos dos dedos
                thumb_tip = points.landmark[4]  # Ponto da ponta do polegar
                x, y = thumb_tip.x, thumb_tip.y

                # Converter a posição normalizada (0-1) para a dimensão da tela
                x_pos = int(x * w)
                y_pos = int(y * h)

                # Mover o cursor do mouse
                pyautogui.moveTo(x_pos * screen_width // w, y_pos * screen_height // h)

        # Mostrar a imagem com os pontos desenhados
        cv2.imshow('Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação dos recursos
video.release()
cv2.destroyAllWindows()
