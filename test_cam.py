# hand_tracking_stage2.py
# Stage 2: Hand tracking + simple gesture detection using MediaPipe
#
# Purpose:
#  - Show hand landmarks on camera feed
#  - Explain landmark coordinates and indexing
#  - Provide a simple, explainable gesture detection function
#
# Usage:
#  python hand_tracking_stage2.py
#  Press ESC to quit.

import cv2
import mediapipe as mp
from math import hypot

# ----------------------------
# MediaPipe setup (explainers)
# ----------------------------
# mp.solutions.hands provides a pipeline that detects hands and returns 21 landmarks.
# Each landmark has normalized coordinates (x,y,z) where x,y are in [0,1] relative to image width/height.
# Landmark indices (0..20) are:
#  0: wrist
#  1..4: thumb (tip = 4)
#  5..8: index (tip = 8)
#  9..12: middle (tip = 12)
# 13..16: ring (tip = 16)
# 17..20: pinky (tip = 20)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Configure the Hands model:
# - max_num_hands: how many hands to detect
# - min_detection_confidence: confidence threshold for initial detection
# - min_tracking_confidence: confidence threshold for landmark tracking (after detection)
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# ----------------------------
# Helper utilities
# ----------------------------
def normalized_to_pixel_coords(landmark, frame_w, frame_h):
    """
    Convert a MediaPipe normalized landmark to pixel coordinates (int).
    landmark: object with .x and .y (normalized)
    frame_w, frame_h: image width and height
    """
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def distance(p1, p2):
    """Euclidean distance between two (x,y) points."""
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----------------------------
# Gesture logic
# ----------------------------
def get_gesture(landmarks, frame_w, frame_h):
    """
    landmarks: list-like of 21 normalized landmarks (MediaPipe's hand.landmark)
    Returns a string with the detected gesture.
    Strategy:
      - Compute finger-up status using tip vs pip (proximal interphalangeal) positions.
      - For thumb, use x-coordinates relative to adjacent landmark because thumb opens sideways.
      - Use additional checks (distances/angles) for thumbs-up and pointing.
    """
    # Convert to pixel coords for more stable comparisons (avoids tiny float issues)
    pts = [normalized_to_pixel_coords(lm, frame_w, frame_h) for lm in landmarks]

    # Indices of tips and pip joints
    tips = [4, 8, 12, 16, 20]         # thumb tip, index tip, middle tip, ring tip, pinky tip
    pip_joints = [2, 6, 10, 14, 18]  # approximations: thumb IP (2), index PIP (6), etc.

    fingers_up = []  # 1 if finger is extended, else 0

    # Thumb: compare x of tip and x of wrist-ish / adjacent depending on handedness.
    # Note: This simple rule assumes camera shows roughly frontal hand. For mirrored camera you might invert.
    wrist_x = pts[0][0]
    thumb_tip_x = pts[tips[0]][0]
    thumb_mcp_x = pts[1][0]  # thumb base
    # If thumb tip is to the right of base and wrist (for right hand) -> extended (this is heuristic)
    if abs(thumb_tip_x - thumb_mcp_x) > 20:  # avoid tiny movements
        # determine direction: if thumb tip is farther from wrist than mcp along x-axis -> extended
        if (thumb_tip_x - wrist_x) * (thumb_mcp_x - wrist_x) > 0:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    else:
        fingers_up.append(0)

    # Other fingers: compare tip.y with pip.y (if tip is above pip -> finger up)
    for i in range(1, 5):
        tip_y = pts[tips[i]][1]
        pip_y = pts[pip_joints[i]][1]
        # smaller y means higher in image coordinates (origin is top-left)
        if tip_y < pip_y - 10:  # -10px margin for stability
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    total_up = sum(fingers_up)

    # Gesture rules (simple and readable)
    # - Fist: no fingers up
    if total_up == 0:
        return "Fist"

    # - Open Palm: all five fingers up
    if total_up == 5:
        return "Open Palm"

    # - Thumbs Up: thumb up but most others down, and thumb tip significantly above wrist in y (pointing upward)
    if fingers_up[0] == 1 and sum(fingers_up[1:]) <= 1:
        # For thumbs up ensure thumb tip y is higher (smaller) than wrist y (thumb pointing up)
        thumb_tip = pts[tips[0]]
        wrist = pts[0]
        if thumb_tip[1] < wrist[1] - 20:  # -20 px tolerance
            return "Thumbs Up"

    # - Pointing (index finger up alone)
    if fingers_up[1] == 1 and sum([fingers_up[i] for i in [0,2,3,4]]) == 0:
        return "Pointing"

    # - Victory (V): index + middle up, others down
    if fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 0 and fingers_up[4] == 0:
        # check separation between index tip and middle tip to confirm 'V' (not touching)
        idx_tip = pts[tips[1]]
        mid_tip = pts[tips[2]]
        if distance(idx_tip, mid_tip) > 40:  # 40px separation heuristic
            return "Victory"

    return "Unknown"

# ----------------------------
# Main loop: capture and process frames
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam. Try changing index to 1 or check permissions.")
        return

    # Optional: change frame size for speed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: empty frame.")
            continue

        # Flip horizontally for a mirror-like view (so gestures feel natural)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe expects RGB images
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture_text = ""

        if results.multi_hand_landmarks:
            # Use first detected hand (we set max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw the landmarks and connections on the frame for visualization
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2))

            # Convert landmarks to a list for easier indexing
            lm_list = hand_landmarks.landmark

            # Compute gesture
            gesture_text = get_gesture(lm_list, w, h)

            # Debug: draw landmark indices near each point (optional)
            # Helpful when learning which index maps to which joint.
            for idx, lm in enumerate(lm_list):
                x_px, y_px = normalized_to_pixel_coords(lm, w, h)
                cv2.putText(frame, str(idx), (x_px+4, y_px-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

        # Overlay gesture text
        if gesture_text:
            # Big text at top-left
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("purthviraj Hands ", frame)

        # ESC to quit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
