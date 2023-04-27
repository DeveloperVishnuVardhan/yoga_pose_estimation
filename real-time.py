"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh
Project: CS-5330 -> Spring 2023.
This file contains the main.py
"""

import cv2
import mediapipe as mp
from utils import get_data_point, load_model
import os


model_path = "Models/model_ANNModel.pth"
pose = mp.solutions.pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
model_path = "Models/model_ANNModel.pth"
loaded_model = load_model(model_path=model_path)
class_names = [name for name in os.listdir(
    "dataset") if os.path.isdir(os.path.join("dataset", name))]


font = cv2.FONT_HERSHEY_SIMPLEX

# Define the text to be displayed
text = "Hello, World!"

# Define the position (x, y) where the text will be displayed
position = (50, 50)

# Define the font scale and thickness
font_scale = 1
font_thickness = 2

# Define the text color (B, G, R) and background color
text_color = (0, 255, 0)  # green text
bg_color = (0, 0, 0)  # Black

cap = cv2.VideoCapture("/Users/jyothivishnuvardhankolla/Desktop/Yoga_pose_estimation/WhatsApp Video 2023-04-26 at 1.53.46 PM.mp4")
while True:
        ret, frame = cap.read()
        if not ret:
             break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = rgb_frame.copy()
        key_points = pose.process(rgb_frame)
        if key_points.pose_landmarks is not None:
            key_points_tensor = get_data_point(key_points.pose_landmarks)
            prediction = loaded_model(key_points_tensor.unsqueeze(0))
            _, class_ = prediction.max(1)
            text = class_names[class_]

            mp_drawing.draw_landmarks(
                annotated_image, key_points.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Display the frame in window.
        bgr_annotated_img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr_annotated_img, text, position, font, font_scale,
                    text_color, font_thickness, cv2.LINE_AA)
        cv2.imshow('Annotated_stream', bgr_annotated_img)

        # Wait for 1 millisecond and check if the user has pressed 'q'.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
