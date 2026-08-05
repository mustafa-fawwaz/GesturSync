import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

filename = "hand_gestures.csv"

# Create/clear the file
with open(filename, mode='w', newline='') as f:
    pass 

cap = cv2.VideoCapture(0)
print("========================================")
print("GESTURE DATA COLLECTOR STARTING...")
print("Hold your hand in the gesture, then press and HOLD the corresponding number key to record.")
print("0: Swipe Left | 1: Swipe Right | 2: Pinch Open | 3: Pinch Close | 4: Neutral Rest")
print("Press 'q' to quit.")
print("========================================")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the 63 features (21 landmarks * x, y, z)
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            cv2.putText(img, "Press 0-4 to Record. 'q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Listen for key presses
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
                label = int(chr(key))
                with open(filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(features + [label])
                print(f"Recorded frame for Class {label}")

    cv2.imshow("Data Collector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()