import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

DATASET_PATH = "hand_gesture_dataset1.csv"

# Create CSV if not exists
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Press any key A-Z to label gesture")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            wrist_x = handLms.landmark[0].x
            wrist_y = handLms.landmark[0].y

            for lm in handLms.landmark:
                landmarks.append(lm.x - wrist_x)
                landmarks.append(lm.y - wrist_y)

    cv2.imshow("Dataset Generator", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if len(landmarks) == 42 and key != -1:
        label = chr(key).upper()
        if label.isalpha():
            with open(DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [label])
            print(f"Saved: {label}")

cap.release()
cv2.destroyAllWindows()
