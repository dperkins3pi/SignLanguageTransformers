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
            list[list[list[list[(str,float,float,float)]]]]: a list of the hand_landmarks for each hand(can be zero) for each frame for each video
        """
        results = []
        for vid_path in video_paths:
            results.append(get_joints(vid_path))
        return results

    def get_joints(vid_path):
        """gets the hand_landmarks of an individual video 

        Args:
            vid_path (str): path to the video to get the hand_landmarks from 

        Returns:
            list[list[list[str,float,float,float]]]]: a list of the hand_landmarks for each hand(can be zero) for each frame
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
            result = detector.detect_from_video(mp_frame, time_stamp)
            hands = []
            for h in result.hand_landmarks:
                points = []
                points.append(("WRIST", h[0].x, h[0].y, h[0].z))
                points.append(("THUM_CMC", h[1].x, h[1].y, h[1].z))
                points.append(("THUMB_MCP", h[2].x, h[2].y, h[2].z))
                points.append(("THUMB_IP", h[3].x, h[3].y, h[3].z))
                points.append(("THUMB_TIP", h[4].x, h[4].y, h[4].z))
                points.append(("INDEX_FINGER_MCP", h[5].x, h[5].y, h[5].z))
                points.append(("INDEX_FINGER_PIP", h[6].x, h[6].y, h[6].z))
                points.append(("INDEX_FINGER_DIP", h[7].x, h[7].y, h[7].z)) 
                points.append(("INDEX_FINGER_TIP", h[8].x, h[8].y, h[8].z))
                points.append(("MIDDLE_FINGER_MCP", h[9].x, h[9].y, h[9].z))
                points.append(("MIDDLE_FINGER_PIP", h[10].x, h[10].y, h[10].z))
                points.append(("MIDDLE_FINGER_DIP", h[11].x, h[11].y, h[11].z))
                points.append(("MIDDLE_FINGER_TIP", h[12].x, h[12].y, h[12].z))
                points.append(("RING_FINGER_MCP", h[13].x, h[13].y, h[13].z))
                points.append(("RING_FINGER_PIP", h[14].x, h[14].y, h[14].z))
                points.append(("RING_FINGER_DIP", h[15].x, h[15].y, h[15].z))
                points.append(("RING_FINGER_TIP", h[16].x, h[16].y, h[16].z))
                points.append(("PINKY_MCP", h[17].x, h[17].y, h[17].z))
                points.append(("PINKY_PIP", h[18].x, h[18].y, h[18].z))
                points.append(("PINKY_DIP", h[19].x, h[19].y, h[19].z))
                points.append(("PINKY_TIP", h[20].x, h[20].y, h[20].z))
                hands.append(points)
            results.append(hands)
        cap.release()
        return results
    