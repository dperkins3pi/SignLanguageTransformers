import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv
from multiprocessing import Pool, cpu_count
from functools import partial

# -1 for all videos (test first with 100 videos)
MAX_VIDEOS_TO_PROCESS = 100
NUM_CORES = cpu_count()

# --- File Paths ---
# MUST BE ABSOLUTE PATHS FOR MULTIPROCESSING
video_path = "/Users/davishunter/Downloads/SignEase/ASL_Citizen/videos"
segmented_folder = "/Users/davishunter/Downloads/SignEase/ASL_Citizen/segmented-videos"
csv_folder = "/Users/davishunter/Downloads/SignEase/ASL_Citizen/joint_data"
os.makedirs(segmented_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# --- Face Mesh Keypoint Selection ---
RELEVANT_FACE_LANDMARKS = [
    # Mouth/Lips (18 points)
    13, 14, 61, 291, 78, 308, 82, 312, 84, 314, 178, 402, 81, 311, 91, 324, 185, 415,
    # Eyes/Lids (8 points)
    33, 133, 144, 153, 263, 362, 373, 382,
    # Eyebrows (6 points)
    52, 55, 105, 282, 285, 334,
    # Stabilization/Context (5 points)
    1, 6, 4, 10, 338
]
NUM_FACE_LANDMARKS = len(RELEVANT_FACE_LANDMARKS)


def get_bounding_box(landmarks, w, h):
    """Calculates a padded bounding box around the hand landmarks."""
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    padding = 20
    return max(0, x_min - padding), max(0, y_min - padding), min(w, x_max + padding), min(h, y_max + padding)


def create_hand_bounding_box_mask(frame, x_min, y_min, x_max, y_max):
    """Creates a mask for the hand bounding box region."""
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


def generate_keypoint_header():
    """Generates the full list of CSV header columns."""
    header = ['frame']

    # Hand Landmarks (21 points per hand * 2 hands)
    for h in range(2):  # hand 0 (right), hand 1 (left)
        for i in range(21):
            for dim in ['x', 'y', 'z']:
                header.append(f'hand_{h}_{i}_{dim}')

    for i in RELEVANT_FACE_LANDMARKS:
        for dim in ['x', 'y', 'z']:
            header.append(f'face_{i}_{dim}')

    return header


def extract_keypoints(frame_num, results_hands, results_face_mesh):
    """
    Extracts and flattens all required landmark coordinates into a single list.
    Missing hands are padded with zeros.
    """
    keypoints = [frame_num]

    # Initialize a buffer for 2 hands (2 * 21 * 3 = 126 values)
    hand_buffer = [0.0] * (2 * 21 * 3)

    if results_hands.multi_hand_landmarks:
        # Determine if the hands are left or right (optional, but good practice)
        handedness = [h.classification[0].label for h in results_hands.multi_handedness]

        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # For simplicity in feature generation, we'll map the first detected hand to buffer index 0,
            # and the second to buffer index 1, regardless of handedness label.
            # If you need consistent Right/Left mapping, more complex logic is needed here.
            hand_idx = i % 2
            start_index = hand_idx * 21 * 3

            for j, lm in enumerate(hand_landmarks.landmark):
                buffer_index = start_index + j * 3
                hand_buffer[buffer_index] = lm.x
                hand_buffer[buffer_index + 1] = lm.y
                hand_buffer[buffer_index + 2] = lm.z

    keypoints.extend(hand_buffer)

    # Face Landmarks (37 points)
    if results_face_mesh.multi_face_landmarks:
        face_landmarks = results_face_mesh.multi_face_landmarks[0]
        for i in RELEVANT_FACE_LANDMARKS:
            lm = face_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        # Pad face landmarks with zeros if no face is detected
        keypoints.extend([0.0] * (NUM_FACE_LANDMARKS * 3))

    return keypoints


def worker_process_video(video_file_name, input_dir, output_dir):
    """
    Worker function for video processing and keypoint extraction.
    """
    # Initialize MediaPipe models INSIDE the worker function (essential)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    input_path = os.path.join(input_dir, video_file_name)

    # Prepare output paths
    base_name = os.path.splitext(video_file_name)[0]
    output_video_path = os.path.join(output_dir, f'segmented-{video_file_name}')
    output_csv_path = os.path.join(csv_folder, f'{base_name}.csv')

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Skipping {video_file_name}: Could not open file.")
        return 0, 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    except Exception as e:
        print(f"[ERROR] Could not initialize VideoWriter for {video_file_name}: {e}")
        cap.release()
        return 0, 0


    start_time = time.time()
    frame_count = 0
    keypoint_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Get points
        results_hands = hands.process(rgb_frame)

        if not results_hands.multi_hand_landmarks:
            continue

        results_face_mesh = face_mesh.process(rgb_frame)

        # 2. Extract and store keypoints for CSV
        keypoints = extract_keypoints(frame_count, results_hands, results_face_mesh)
        keypoint_data.append(keypoints)

        # 3. Core Segmentation and Video Output (unchanged logic)
        hand_face_regions_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hand_bounding_boxes = []
        face_landmarks_list = []

        for hand_landmarks in results_hands.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h)
            hand_bounding_boxes.append((x_min, y_min, x_max, y_max, hand_landmarks))
            hand_bbox_mask = create_hand_bounding_box_mask(frame, x_min, y_min, x_max, y_max)
            hand_face_regions_mask = cv2.bitwise_or(hand_face_regions_mask, hand_bbox_mask)

        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                face_landmarks_list.append(face_landmarks)
                face_outline_mask = create_face_mask_from_landmarks(frame, face_landmarks, w, h)
                hand_face_regions_mask = cv2.bitwise_or(hand_face_regions_mask, face_outline_mask)

        final_mask = hand_face_regions_mask
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)

        segmented_frame = cv2.bitwise_and(frame, frame, mask=final_mask)
        grayscale_result = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)

        output_frame = cv2.cvtColor(grayscale_result, cv2.COLOR_GRAY2BGR)
        out.write(output_frame)

    # Cleanup models
    cap.release()
    out.release()
    hands.close()
    face_mesh.close()

    # 4. Write CSV File
    if keypoint_data:
        csv_header = generate_keypoint_header()
        try:
            with open(output_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(csv_header)
                writer.writerows(keypoint_data)
        except Exception as e:
            print(f"[ERROR] Could not write CSV for {video_file_name}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    return frame_count, elapsed_time


def main():
    print(f"--- Batch Video Processing Started ---")

    # 1. Identify videos to process
    all_files = os.listdir(video_path)
    video_files = [f for f in all_files if
                   f.lower().endswith(('.mp4', '.avi', '.mov')) and os.path.isfile(os.path.join(video_path, f))]

    videos_to_process = video_files[:MAX_VIDEOS_TO_PROCESS]

    if not videos_to_process:
        print(f"Error: No video files found or none to process.")
        return

    print(f"Total video files found: {len(video_files)}. Processing {len(videos_to_process)} videos using {NUM_CORES} cores.")

    start_batch_time = time.time()

    # 2. Setup multiprocessing pool
    worker_func_partial = partial(worker_process_video,
                                  input_dir=video_path,
                                  output_dir=segmented_folder)

    # Use Pool to map the worker function across the list of video file names
    with Pool(NUM_CORES) as pool:
        results = pool.map(worker_func_partial, videos_to_process)

    end_batch_time = time.time()

    # 3. Aggregate results
    valid_results = [res for res in results if res != (0, 0)]
    total_frames = sum(res[0] for res in valid_results)

    total_wall_time = end_batch_time - start_batch_time

    # --- Final Summary ---
    processed_count = len(valid_results)
    avg_fps = total_frames / total_wall_time if total_wall_time > 0 else 0

    print(f"\n========================================================")
    print(f"BATCH PROCESSING COMPLETE: {processed_count} of {len(videos_to_process)} videos successfully processed.")
    print(f"Total Wall-Clock Time: {total_wall_time:.2f} seconds.")
    print(f"Total Frames Processed: {total_frames}.")
    print(f"Overall Processing Rate: {avg_fps:.2f} FPS (Wall-Clock).")
    print(f"Output saved to: {segmented_folder}")
    print(f"========================================================")


if __name__ == '__main__':
    main()