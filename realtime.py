import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ✅ Must match the order used in training (sorted)
gesture_labels = [
    "choe", "goodbye", "hello", "I love you", "juthrin",
    "kuzu", "nga", "no", "please", "thank you", "yes"
]

# Load model
model = load_model('ultimate_model.keras')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
sequence = []
seq_length = 30
display_threshold = 0.7
last_prediction = ''
last_confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    keypoints = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks[:2]:
            single_hand = []
            for lm in hand_landmarks.landmark:
                single_hand.extend([lm.x, lm.y, lm.z])
            keypoints.extend(single_hand)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(result.multi_hand_landmarks) == 1:
            keypoints.extend([0.0] * 63)  # pad second hand

        if len(keypoints) == 126:
            sequence.append(keypoints)
            if len(sequence) > seq_length:
                sequence.pop(0)

        if len(sequence) == seq_length:
            input_seq = np.expand_dims(sequence, axis=0)  # (1, 30, 126)
            preds = model.predict(input_seq, verbose=0)[0]
            pred_index = np.argmax(preds)
            confidence = preds[pred_index]

            if confidence > display_threshold and pred_index < len(gesture_labels):
                last_prediction = gesture_labels[pred_index]
                last_confidence = confidence
    else:
        sequence = []  # reset if no hand detected

    # Show prediction
    cv2.putText(frame, f'Prediction: {last_prediction} ({last_confidence:.2f})',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-Time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
