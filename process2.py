import os
import cv2
import numpy as np
import mediapipe as mp

# Constants
SEQUENCE_LENGTH = 30
DATA_DIR = 'data'
FEATURES_PER_HAND = 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # Max for 2 hands

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Alphabetically sorted gestures
all_gestures = sorted([
    "choe", "goodbye", "hello", "I love you", "juthrin",
    "kuzu", "nga", "no", "please", "thank you", "yes"
])
print("Expected gestures:", all_gestures)

X_seq = []
y_seq = []

for label_idx, gesture in enumerate(all_gestures):
    gesture_path = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(gesture_path):
        print(f"⚠️ Gesture folder not found: {gesture_path}")
        continue

    images = sorted([f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.png'))])
    print(f"📂 Processing {len(images)} frames for gesture '{gesture}'")

    sequence = []
    for img_file in images:
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to load {img_path}, skipping")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            keypoints = []
            hand_data = []

            for hand_landmarks in result.multi_hand_landmarks[:2]:  # Max 2 hands
                single_hand = []
                for lm in hand_landmarks.landmark:
                    single_hand.extend([lm.x, lm.y, lm.z])
                hand_data.append(single_hand)

            # Padding logic
            if len(hand_data) == 1:
                hand_data.append([0.0] * FEATURES_PER_HAND)

            # Flatten both hands into one list (length = 126)
            keypoints = hand_data[0] + hand_data[1]
            sequence.append(keypoints)

            if len(sequence) == SEQUENCE_LENGTH:
                X_seq.append(sequence)
                y_seq.append(label_idx)
                sequence = []
        else:
            print(f"🖐️ No hands detected in {img_path}, skipping")

# Convert to numpy arrays
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"\n✅ Final data shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
np.save('X_seq.npy', X_seq)
np.save('y_seq.npy', y_seq)
print("💾 Saved sequence data to X_seq.npy and y_seq.npy")

hands.close()
