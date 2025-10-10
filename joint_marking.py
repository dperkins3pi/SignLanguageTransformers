import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib import pyplot as plt
import numpy as np
import cv2
import requests

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

response = requests.get(MODEL_URL)
with open(MODEL_PATH, 'wb') as f:
    f.write(response.content)
    
def draw_joint_markers(vid, result):
    return None

def get_joints_all(video_paths):
    """gets the joints of all videos in a list of videos

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
        possible alternative instead return a video with the markings. (preferred eventual behavior)
    """
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(running_mode=vision.RunningMode.VIDEO, base_options=base_options, num_hands=2) #Currently will detect a max of 2 hands.
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_per_ms = 1000.0 / fps
    frame_num = 0
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
        frame_num += 1
    return results  