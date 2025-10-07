import cv2
import mediapipe as mp
import pyautogui
import time

# Mediapipe Hand Tracking Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Webcam feed
cap = cv2.VideoCapture(1)

# Gesture thresholds and state variables
dragging = False
drag_start_time = None
drag_hold_time = 0.2  # Minimum time for a drag action
click_threshold = 0.02  # Distance to detect a click
smooth_pointer_x, smooth_pointer_y = None, None

# Helper to scale normalized coordinates to screen dimensions
def scale_to_screen(normalized_x, normalized_y, frame_width, frame_height):
    x = min(max(normalized_x, 0), 1)
    y = min(max(normalized_y, 0), 1)
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    return screen_x, screen_y

# Main loop for gesture control
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for mirror effect and process with Mediapipe
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Scale landmarks to screen
            pointer_x, pointer_y = scale_to_screen(index_tip.x, index_tip.y, frame_width, frame_height)

            # Stabilize pointer movement
            if smooth_pointer_x is None or smooth_pointer_y is None:
                smooth_pointer_x, smooth_pointer_y = pointer_x, pointer_y
            else:
                smooth_pointer_x = int(0.7 * smooth_pointer_x + 0.3 * pointer_x)
                smooth_pointer_y = int(0.7 * smooth_pointer_y + 0.3 * pointer_y)

            # Move the pointer
            pyautogui.moveTo(smooth_pointer_x, smooth_pointer_y)

            # Calculate distances for gestures
            index_thumb_distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5
            index_down_distance = index_tip.y - index_pip.y
            middle_down_distance = middle_tip.y - index_pip.y

            # Left Click: Triggered when index finger tip moves down
            if index_down_distance > 0.05:
                pyautogui.click()
                time.sleep(0.2)  # Small delay to avoid multiple clicks
                print("Left Click")

            # Right Click: Triggered when middle finger tip moves down
            elif middle_down_distance > 0.05:
                pyautogui.rightClick()
                time.sleep(0.2)  # Small delay to avoid multiple clicks
                print("Right Click")

            # Drag-and-Drop / Selection: Triggered when index and thumb tips touch
            if index_thumb_distance < click_threshold:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    print("Drag Start")
            elif dragging:
                pyautogui.mouseUp()
                dragging = False
                print("Drag End")

    # Display the webcam feed
    cv2.imshow("Hand Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
