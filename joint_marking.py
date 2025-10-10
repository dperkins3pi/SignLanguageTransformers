import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
import requests

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

response = requests.get(MODEL_URL)
with open(MODEL_PATH, 'wb') as f:
    f.write(response.content)
    
    # def draw_joint_markers(frame, result):
    #     """a method to draw the joints on hands in an image

    #     Args:
    #         frame (_type_): _description_
    #         result (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     MARGIN = 10  # pixels
    #     FONT_SIZE = 1
    #     FONT_THICKNESS = 1
    #     HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    #     hand_landmarks_list = result.hand_landmarks
    #     handedness_list = result.handedness
    #     annotated_image = np.copy(frame)
    #     # Loop through the detected hands to visualize.
    #     for idx in range(len(hand_landmarks_list)):
    #         hand_landmarks = hand_landmarks_list[idx]
    #         handedness = handedness_list[idx]

    #         # Draw the hand landmarks.
    #         hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #         hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
    #         solutions.drawing_utils.draw_landmarks(
    #         annotated_image,
    #         hand_landmarks_proto,
    #         solutions.hands.HAND_CONNECTIONS,
    #         solutions.drawing_styles.get_default_hand_landmarks_style(),
    #         solutions.drawing_styles.get_default_hand_connections_style())

    #         # Get the top left corner of the detected hand's bounding box.
    #         height, width, _ = annotated_image.shape
    #         x_coordinates = [landmark.x for landmark in hand_landmarks]
    #         y_coordinates = [landmark.y for landmark in hand_landmarks]
    #         text_x = int(min(x_coordinates) * width)
    #         text_y = int(min(y_coordinates) * height) - MARGIN
    #     return annotated_image

    def get_joints_all(video_paths):
        """currently gets the joints of all videos in a list of videos.
        possible alternative instead return a video with the markings.

        Args:
            video_paths (list[str]): a list of filepaths to videos

        Returns:
            list[list[HandLandmarkerResult]]: a list of lists of HandLandmarkerResults where each list[HandLandmarkerResults corresponds to a given video]
        """
        results = []
        for vid_path in video_paths:
            results.append(get_joints(vid_path))
        return results

    def get_joints(vid_path):
        """gets the HandLandmarkerResults of an individual video 

        Args:
            vid_path (str): path to the video to get the HandLandmarkerResults from 

        Returns:
            list[HandLandmarkerResults]: a list of the HandLandmarkerResults for each video frame. 
        """
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(running_mode=vision.RunningMode.VIDEO, base_options=base_options, num_hands=2) #Currently will detect a max of 2 hands.
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_per_ms = 1000.0 / fps
        frame_num = 0
        # possible test via drawing on a single video
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # new_vid = cv2.VideoWriter('output_edited.mp4', fourcc, fps, (width, height))
        results = []
        with vision.HandLandmarker.create_from_options(options) as detector:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            time_stamp = frame_per_ms * frame_num
            results.append(detector.detect_from_video(mp_frame, time_stamp))
        cap.release()
        return results