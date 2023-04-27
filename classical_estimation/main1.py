"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh
Project: CS-5330 -> Spring 2023.
This file contains the main1.py
"""

import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
from instructor import *
from utils import *
import matplotlib.pyplot as plt
from IPython.display import HTML


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

modelType = int(argv[1])
# Initialize the Videocapture object to read from the webcan.
video = cv2. VideoCapture(0)
# creute named window for resizing purposes
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
# Inittatize the videocapture object to read from a video stored in the disk.
#video = cv2. Videocopture('redia/running. me*)

#Set video camera size
video.set(3,1280)
video.set (4,960)

# itialize a variable to store the tire of the previous from
time1 = 0

# Iterate until the video is accessed successfully,
while video.isOpened():

    # Read a frame
    ok, frame = video.read()
    # Check if from
    if not ok:
        break
    # Flip the frone hortzontally for natural (selfie-vie
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ = frame.shape
    
    frame = cv2.resize(frame, (int(frame_width * (640/frame_height)), 600))
    frame, _ = classifyPose(landmarks, lm, frame, display=False)


    if landmarks:
        lm = getLandmarkAngles(landmarks)
        if modelType == 0:
            frame, _ = classifyPose(landmarks, lm, frame, display=False)
            print("right")

        elif modelType == 1:
            print("right")
            frame, _ = warriorPose(landmarks, lm,  frame, display=False)
        else:
            frame, _ = warriorPose(landmarks, lm,  frame, display=False)

    #frame, _ = detectPose(frame, pose_video, display=False)
    frame, landmarks = detectPose(frame, pose_video, display=False)

    
    time2 = time()
    
    if (time2 - time1) > 0:
        frames_per_second = 1.0/(time2 - time1)
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
    time1 = time2
    cv2.imshow('Pose Classification', frame)
    k = cv2.waitKey(1) & 0xFF
    if (k ==27):
        break
video.release()
cv2.destroyAllWindows()

