import cv2
import os
import time
import mediapipe as mp

# List of gesture labels
gestures = ["hello", "thank you", "yes", "no", "please", "I love you", "goodbye"]

# Create a folder for each gesture
for gesture in gestures:
    os.makedirs(gesture, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

gesture_index = 0
gesture = gestures[gesture_index]
sample_count = 0
collecting = False
frames_collected = 0

print("Press SPACE to start recording a sample (30 frames per sample).")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay text on frame
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Samples: {sample_count}/100", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if collecting:
        cv2.putText(frame, f"Frames: {frames_collected}/30", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit program
    if key == ord('q'):
        print("Exiting...")
        break

    # Start countdown and begin collecting 30 frames on spacebar press
    if key == ord(' ') and not collecting and sample_count < 100:
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Starting in {i}", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1000)
        print(f"Recording sample {sample_count + 1}/100 for '{gesture}'...")
        collecting = True
        frames_collected = 0

    # Collect 30 frames (1 second @ 30 FPS)
    if collecting:
        if frames_collected < 30:
            save_path = os.path.join(gesture, f"{gesture}_{sample_count}_{frames_collected}.jpg")
            cv2.imwrite(save_path, frame)
            frames_collected += 1
            time.sleep(1 / 30)  # Maintain 30 FPS
        else:
            collecting = False
            sample_count += 1
            print(f"✅ Sample {sample_count}/100 collected for '{gesture}'")

            if sample_count == 100:
                gesture_index += 1
                if gesture_index < len(gestures):
                    gesture = gestures[gesture_index]
                    sample_count = 0
                    print(f"\n➡️ Next gesture: {gesture}")
                else:
                    print("✅ All gestures completed.")
                    break

cap.release()
cv2.destroyAllWindows()
