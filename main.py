import cv2
import mediapipe as mp
import math
import vgamepad as vg
from pynput.keyboard import Controller as KController, Key

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,model_complexity=0, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Optimize frame size and reduce camera buffer lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Keeps frame buffer fresh
cv2.setUseOptimized(True)
# Initialize virtual gamepad
try:
    gamepad = vg.VX360Gamepad()
    gamepad_ready = True
    print("Virtual gamepad initialized successfully")
except Exception as e:
    gamepad_ready = False
    print(f"Warning: Could not initialize gamepad: {e}")
    print("Make sure ViGEmBus is installed. Continuing without gamepad...")
if not cap.isOpened(): raise SystemExit("Cannot open camera")
flip = True
last_hand_count = -1
smooth_x = 0.0
alpha = 0.4
steering = 0.0
analoge = True
left_pressed = False
right_pressed = False
throttle_pressed = False
brake_pressed = False
keyboard = KController()


def is_fist(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    knuckle = hand_landmarks.landmark[9]  # Middle knuckle
    tip = hand_landmarks.landmark[12]  # Middle fingertip

    palm_len = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y)
    tip_len = math.hypot(tip.x - wrist.x, tip.y - wrist.y)

    if palm_len == 0:
        return False

    ratio = tip_len / palm_len
    return ratio < 1.50  # Scale-invariant ratio

while True:
    ret, frame = cap.read()
    if not ret: break
    if flip:
        frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    current_hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    h, w, _ = frame.shape
    knuckle_points = []

    # Extract knuckles from all detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[0]
            knuckle = hand_landmarks.landmark[9]
            tip = hand_landmarks.landmark[12]
            palm_len = math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y)
            tip_len = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            ratio = tip_len / palm_len if palm_len > 0 else 0.0
            knuckle_x_px = int(knuckle.x * w)
            knuckle_y_px = int(knuckle.y * h)
            knuckle_points.append((knuckle_x_px, knuckle_y_px))
            cv2.circle(frame, (knuckle_x_px, knuckle_y_px), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"R:{ratio:.2f}", (knuckle_x_px - 20, knuckle_y_px - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Only calculate steering if exactly 2 hands detected
    # Only calculate steering if exactly 2 knuckle points detected
    if len(knuckle_points) == 2:
        # Ensure consistent ordering: left point first, right point second
        left_pt, right_pt = sorted(knuckle_points, key=lambda p: p[0])
        x1, y1 = left_pt
        x2, y2 = right_pt

        # Visual: connect left->right
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        dx = x2 - x1  # always >= 0
        dy = y2 - y1

        # Angle from horizontal (degrees)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        MAX_ANGLE = 80.0
        angle_offset = 0.0
        raw_steering = (angle_deg - angle_offset) / MAX_ANGLE
        raw_steering = max(-1.0, min(1.0, raw_steering))

        # Optional: apply a small deadzone and scale for symmetry & stability
        deadzone = 0.03  # ignore tiny motions
        scale = 0.98  # slightly compress full range if needed
        s = raw_steering * scale
        if abs(s) < deadzone:
            s = 0.0

        # Smooth the steering value
        steering = alpha * s + (1 - alpha) * steering


    else:
        steering = 0.0
        cv2.putText(frame, "Show 2 hands to steer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if (analoge):
        cv2.putText(frame, f"Mode: Analogue", (470, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, f"Mode: Discrete", (470, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # Send steering to virtual gamepad (with debug axis calculation)
    if gamepad_ready:
        # Map steering (-1..1) to gamepad X-axis (0..65535)
        axis_value = int((steering + 1.0) / 2.0 * 65535)

        if len(knuckle_points) == 2 and results.multi_hand_landmarks:
            # --- CALCULATE 3-STATE PEDAL LOGIC ---
            fist_count = sum(1 for hand in results.multi_hand_landmarks if is_fist(hand))

            is_gas = (fist_count == 2)  # Both fists = Gas
            is_brake = (fist_count == 0)  # Both open = Brake

            if analoge:
                gamepad.left_joystick(axis_value, 32767)
                if left_pressed:
                    keyboard.release(Key.left)
                    left_pressed = False
                if right_pressed:
                    keyboard.release(Key.right)
                    right_pressed = False
                gamepad.right_trigger(255 if is_gas else 0)
                gamepad.left_trigger(255 if is_brake else 0)
            else:
                if (raw_steering > 0.2):
                    if not right_pressed:
                        keyboard.press(Key.right)
                        right_pressed = True
                        print("Right pressed")
                    if left_pressed:
                        keyboard.release(Key.left)
                        left_pressed = False
                if (raw_steering < -0.2):
                    if not left_pressed:
                        keyboard.press(Key.left)
                        left_pressed = True
                        print("Left pressed")
                    if right_pressed:
                        keyboard.release(Key.right)
                        right_pressed = False
                if (raw_steering <0.2 and raw_steering > -0.2):
                    if left_pressed:
                        keyboard.release(Key.left)
                        left_pressed = False
                    if right_pressed:
                        keyboard.release(Key.right)
                        right_pressed = False
                if is_gas:
                    if not throttle_pressed:
                        keyboard.press(Key.up);
                        throttle_pressed = True
                    if brake_pressed:
                        keyboard.release(Key.down);
                        brake_pressed = False
                elif is_brake:
                    if not brake_pressed:
                        keyboard.press(Key.down);
                        brake_pressed = True
                    if throttle_pressed:
                        keyboard.release(Key.up);
                        throttle_pressed = False
                else:  # Neutral (1 fist)
                    if throttle_pressed:
                        keyboard.release(Key.up);
                        throttle_pressed = False
                    if brake_pressed:
                        keyboard.release(Key.down);
                        brake_pressed = False

            gamepad.update()

        else:
            # keep centered when hands not present
            gamepad.left_joystick(32767, 32767)
            if left_pressed:
                keyboard.release(Key.left)
                left_pressed = False
            if right_pressed:
                keyboard.release(Key.right)
                right_pressed = False
            if throttle_pressed:
                keyboard.release(Key.up)
                throttle_pressed = False
            if brake_pressed:
                keyboard.release(Key.down)
                brake_pressed = False
            gamepad.update()


    # Draw steering bar
    bar_w = int(w * 0.8)
    bar_h = 20
    bar_x = int((w - bar_w) / 2)
    bar_y = h - 40
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    pos = int((steering + 1) / 2 * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + pos, bar_y + bar_h), (0, 180, 255), -1)
    cv2.putText(frame, f"Steer: {steering:.2f}", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if current_hand_count != last_hand_count:
        print("Hands:", current_hand_count)
        last_hand_count = current_hand_count
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quit")
        break
    if key == ord('f'):
        flip = not flip
        print("Camera Flipped")
    if key == ord('a'):
        analoge = not analoge
        print("Discrete/Analogue mode toggled")
cap.release()
cv2.destroyAllWindows()
keyboard.release(Key.up)