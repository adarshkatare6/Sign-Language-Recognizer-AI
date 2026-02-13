
import gradio as gr
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter

# Load model and scaler
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

buffer = deque(maxlen=15)
current_word = ""
current_char = ""

def predict_stream(image):
    global current_word, current_char

    if image is None:
        return current_word, current_char, 0.0

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        landmarks = np.array(landmarks).reshape(1, -1)
        landmarks = scaler.transform(landmarks)

        probs = model.predict_proba(landmarks)[0]
        pred = model.classes_[np.argmax(probs)]
        confidence = np.max(probs)

        buffer.append(pred)
        current_char = pred

        if len(buffer) == 15:
            most_common = Counter(buffer).most_common(1)[0][0]
            count = Counter(buffer)[most_common]

            if count > 12:
                if len(current_word) == 0 or current_word[-1] != most_common:
                    current_word += most_common

        return current_word, current_char, float(confidence)

    return current_word, "-", 0.0


def clear_text():
    global current_word
    current_word = ""
    return "", "-", 0.0


# ---------------- UI ---------------- #

with gr.Blocks(theme=gr.themes.Soft(), title="Sign Language AI") as demo:

    gr.Markdown(
        """
        # Sign Language Recognition System  
        ## Real-time Hand Gesture to Text Conversion  
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            camera = gr.Image(sources=["webcam"], streaming=True, label="Live Camera")

        with gr.Column(scale=1):
            word_output = gr.Textbox(label="-> Predicted Word", lines=2)
            char_output = gr.Textbox(label="->  Current Character")
            confidence_output = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                label="📊 Confidence Score"
            )
            clear_btn = gr.Button("Clear Text", variant="stop")

    camera.stream(
        predict_stream,
        inputs=camera,
        outputs=[word_output, char_output, confidence_output]
    )

    clear_btn.click(
        clear_text,
        outputs=[word_output, char_output, confidence_output]
    )

demo.launch()
