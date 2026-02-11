import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

# Load model & scaler
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=15)
current_word = ""
last_char = ""
last_time = time.time()

def extract_features(handLms):
    wrist_x = handLms.landmark[0].x
    wrist_y = handLms.landmark[0].y

    scale = np.sqrt(
        (handLms.landmark[9].x - wrist_x)**2 +
        (handLms.landmark[9].y - wrist_y)**2
    ) + 1e-6

    landmarks = []
    for lm in handLms.landmark:
        landmarks.append((lm.x - wrist_x) / scale)
        landmarks.append((lm.y - wrist_y) / scale)

    return np.array(landmarks).reshape(1, -1)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    predicted_char = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label

            # Only right hand
            if hand_label != "Right":
                continue

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            X = extract_features(handLms)
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]

            prediction_buffer.append(pred)
            predicted_char = Counter(prediction_buffer).most_common(1)[0][0]

            # Character commit logic (time-based debounce)
            if predicted_char != last_char and time.time() - last_time > 1.5:
                current_word += predicted_char
                last_char = predicted_char
                last_time = time.time()

    cv2.putText(img, f"Char: {predicted_char}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img, f"Word: {current_word}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Sign Language Recognition", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):   # space = end word
        current_word += " "
    elif key == ord('c'):   # clear
        current_word = ""

cap.release()
cv2.destroyAllWindows()
