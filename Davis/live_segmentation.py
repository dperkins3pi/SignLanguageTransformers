import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
SHOW_COLORED_JOINTS = True
# ---------------------

# --- Face Mesh Keypoint Selection ---
RELEVANT_FACE_LANDMARKS = [
    # --- Mouth/Lips (Minimal 18 points for shape, corner, and center dynamics) ---
    13, 14, # Outer Lips (Top/Bottom Center)
    61, 291, # Inner Lips (Top/Bottom Center)
    78, 308, # Outer Lips (Left/Right Center)
    82, 312, # Inner Lips (Left/Right Center)
    84, 314, # Left/Right Corner
    178, 402, # Top Lip Contour (Upper)
    81, 311, # Bottom Lip Contour (Lower)
    91, 324, # Upper Lip (Inner Edge)
    185, 415, # Lower Lip (Inner Edge)

    # --- Eyes/Lids (Minimal 8 points: Corners, Centers, and Eyelid dynamics) ---
    # Right Eye (4 points)
    33, 133,  # Inner/Outer Corner
    144, 153, # Upper/Lower Lid Center
    # Left Eye (4 points)
    263, 362, # Inner/Outer Corner
    373, 382, # Upper/Lower Lid Center

    # --- Eyebrows
    # Right Eyebrow (3 points)
    52,  # Inner Edge (Near Glabella)
    55,  # Center/Arch Peak
    105,  # Outer Edge
    # Left Eyebrow (3 points)
    282,  # Inner Edge (Near Glabella)
    285,  # Center/Arch Peak
    334,  # Outer Edge

    # --- Stabilization/Context (Minimal 5 points: Face/Head orientation) ---
    1, # Nose Tip (Primary stabilization)
    6, # Nose Base
    4, # Chin (Vertical reference)
    10, # Right Cheek (Lateral reference)
    338 # Left Cheek (Lateral reference)
]

# Total landmarks reduced from 73 to 35.
NUM_FACE_LANDMARKS = len(RELEVANT_FACE_LANDMARKS)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
# Use static_image_mode=False for better tracking continuity in video
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


# Removed: mp_selfie_segmentation and selfie_segmentation initialization

# Removed: get_hand_color_range and create_color_mask functions
# The following helper functions are kept as they are needed.

def get_bounding_box(landmarks, w, h):
    """Calculates a padded bounding box around the hand landmarks."""
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    padding = 20
    return max(0, x_min - padding), max(0, y_min - padding), min(w, x_max + padding), min(h, y_max + padding)


def create_hand_bounding_box_mask(frame, x_min, y_min, x_max, y_max):
    """Creates a mask for the hand bounding box region (now the ONLY hand mask)."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def create_face_mask_from_landmarks(frame, face_landmarks, w, h):
    """
    Creates a mask for the face region based on the convex hull of ALL 468 face mesh landmarks.
    """
    points = []
    for landmark in face_landmarks.landmark:
        x = min(int(landmark.x * w), w - 1)
        y = min(int(landmark.y * h), h - 1)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def main_live_feed():
    """Main loop for live webcam processing and display."""

    # Use camera 0 (default)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set initial global state for visualization toggle
    global SHOW_COLORED_JOINTS

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Get face and hand points
        results_hands = hands.process(rgb_frame)
        results_face_mesh = face_mesh.process(rgb_frame)

        # Removed Step 2: Selfie segmentation is completely removed.

        # 2. Create masks for hand (via Bounding Box) and face regions (via Convex Hull)
        hand_face_regions_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hand_bounding_boxes = []
        face_landmarks_list = []

        # Process hands (using Bounding Box ONLY)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h)
                hand_bounding_boxes.append((x_min, y_min, x_max, y_max, hand_landmarks))

                # Now always use the Bounding Box mask
                hand_bbox_mask = create_hand_bounding_box_mask(frame, x_min, y_min, x_max, y_max)
                hand_face_regions_mask = cv2.bitwise_or(hand_face_regions_mask, hand_bbox_mask)

        # Process face (using Convex Hull)
        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                face_landmarks_list.append(face_landmarks)
                # NOTE: create_face_mask_from_landmarks uses ALL 468 points for robust face segmentation
                face_outline_mask = create_face_mask_from_landmarks(frame, face_landmarks, w, h)
                hand_face_regions_mask = cv2.bitwise_or(hand_face_regions_mask, face_outline_mask)

        # 3. Final Segmentation Mask: ONLY the combined hand and face regions mask
        final_mask = hand_face_regions_mask

        # Apply morphological operations for mask cleanup
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        # 4. Apply mask and convert to grayscale
        segmented_frame = cv2.bitwise_and(frame, frame, mask=final_mask)
        grayscale_result = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)

        # 5. Prepare frame for output display (Visualization remains the same)
        if SHOW_COLORED_JOINTS:
            # Visualization output (BGR)
            display_frame = cv2.cvtColor(grayscale_result, cv2.COLOR_GRAY2BGR)

            # Draw colored landmarks/bboxes

            # Hands: Bounding box and full landmarks
            for x_min, y_min, x_max, y_max, hand_landmarks in hand_bounding_boxes:
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

            # Face: Highlight RELEVANT_FACE_LANDMARKS
            for face_landmarks in face_landmarks_list:
                # Draw small connections for facial context (optional, can be removed if too cluttered)
                mp_drawing.draw_landmarks(
                    display_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=0),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
                )

                # Draw distinct circles for the 73 relevant landmarks
                for index in RELEVANT_FACE_LANDMARKS:
                    lm = face_landmarks.landmark[index]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # Use a prominent red circle (BGR: 0, 0, 255)
                    cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)

            cv2.imshow('Live Segmentation - Press C to toggle visualization, Q to quit', display_frame)
        else:
            cv2.imshow('Live Segmentation - Press C to toggle visualization, Q to quit', grayscale_result)

        # Key press handler
        key = cv2.waitKey(5) & 0xFF
        if key == ord('c'):
            SHOW_COLORED_JOINTS = not SHOW_COLORED_JOINTS
            print(f"Colored joints visualization: {'ON' if SHOW_COLORED_JOINTS else 'OFF'}")
        elif key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_live_feed()