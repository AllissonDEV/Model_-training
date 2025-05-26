import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

MODEL_PATH = ...
with open('labels.txt', 'r') as file:
    CLASS_NAMES = [line.strip() for line in file.readlines()]

hand_modulo = mp.solutions.hands

hands = hand_modulo.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def process_frame(frame):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, hand_modulo.HAND_CONNECTIONS)
            x_max, y_max = 0, 0
            x_min, y_min = frame.shape[1], frame.shape[0]

            for lm in hand.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_max, x_min = max(x, x_max), min(x, x_min)
                y_max, y_min = max(y, y_max), min(y, y_min)

            margin = 50
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(frame.shape[1], x_max + margin), min(frame.shape[0], y_max + margin)
            img_crop = frame[y_min:y_max, x_min:x_max]

            if img_crop.size == 0:
                continue

            cv2.imshow("Recorte", img_crop)
            img_crop = cv2.resize(img_crop, (224, 224))
            img_array = np.asarray(img_crop, dtype=np.float32)
            normalized_image = (img_array / 127.0) - 1
            data = np.expand_dims(normalized_image, axis=0)
            prediction = model.predict(data, verbose=0)
            return CLASS_NAMES[np.argmax(prediction)]
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    result = process_frame(img)
    if result:
        cv2.putText(img, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('esc'):
        break

cap.release()
cv2.destroyAllWindows()
