import cv2
import mediapipe as mp
import pyautogui
import math
import time
import joblib
import numpy as np

# ==========================================
# 1. Initialization
# ==========================================
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- LOAD YOUR TRAINED AI MODEL ---
try:
    print("Loading SVM Model...")
    svm_model = joblib.load('gestursync_svm_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: gestursync_svm_model.pkl not found! Please run train_svm.py first.")
    exit()

# Class mapping based on your data collector
class_names = ["Swipe Left", "Swipe Right", "Pinch Open", "Pinch Close", "Neutral"]

cooldown_swipe = 1.0
cooldown_vol = 0.1
cooldown_mute = 1.0

last_swipe_time = time.time()
last_vol_time = time.time()
last_mute_time = time.time()

# ==========================================
# 2. Main Video Loop
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("GesturSync AI Live starting... Press 'q' to quit.")

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the real image to find hands
    results = hands.process(img_rgb)

    # OVERWRITE the image with a black canvas before drawing anything
    img = np.zeros_like(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = img.shape

            # --- EXTRACT ALL 63 LANDMARKS FOR THE SVM ---
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            
            # Predict the gesture using your trained model
            feature_array = np.array(features).reshape(1, -1)
            predicted_class = svm_model.predict(feature_array)[0]
            current_gesture = class_names[predicted_class]

            # Display the AI's thought process on screen
            cv2.putText(img, f"AI: {current_gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # --- EXTRACT SPECIFIC LANDMARKS FOR HEURISTICS (Volume/Mute) ---
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            pinky_tip = hand_landmarks.landmark[20]

            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            px, py = int(pinky_tip.x * w), int(pinky_tip.y * h)

            current_time = time.time()

            # ==========================================
            # FEATURE 1: Swipes via SVM Model
            # ==========================================
            if current_time - last_swipe_time > cooldown_swipe:
                if predicted_class == 0:  # AI detected Swipe Left
                    pyautogui.press('left')
                    cv2.putText(img, "PREV SLIDE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    last_swipe_time = current_time
                elif predicted_class == 1:  # AI detected Swipe Right
                    pyautogui.press('right')
                    cv2.putText(img, "NEXT SLIDE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    last_swipe_time = current_time

            # ==========================================
            # FEATURE 2: Instant Mute (Heuristic: Thumb to Pinky)
            # ==========================================
            mute_distance = math.hypot(px - tx, py - ty)
            if current_time - last_mute_time > cooldown_mute:
                if mute_distance < 30:
                    pyautogui.press('volumemute')
                    cv2.putText(img, "MUTED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    last_mute_time = current_time

            # ==========================================
            # FEATURE 3: Dynamic Volume (Heuristic: Thumb to Index)
            # ==========================================
            # Only trigger volume heuristics if the AI isn't detecting a specific swipe or pinch
            if predicted_class == 4: # Neutral Rest
                vol_distance = math.hypot(ix - tx, iy - ty)
                cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 3)

                if current_time - last_vol_time > cooldown_vol:
                    if vol_distance < 30:
                        pyautogui.press('volumedown')
                        cv2.putText(img, "Vol DOWN", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        last_vol_time = current_time
                    elif vol_distance > 120:
                        pyautogui.press('volumeup')
                        cv2.putText(img, "Vol UP", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        last_vol_time = current_time

    cv2.putText(img, "GesturSync V2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("GesturSync", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
