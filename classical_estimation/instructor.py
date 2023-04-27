
"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh
Project: CS-5330 -> Spring 2023.
This file contains instructormodel
"""

import math
import cv2
import numpy as np
from time import time
from utils import *
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import HTML
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def getLandmarkAngles(landmarks):
    '''
    This function gets all the key landmarks angles which is shoulder, hip, knee, elbow for both left and right sides
    Args:
    Returns:
        output:
    '''
     # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
       # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    # Get angle between thigs and hip 12 24 26
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    # 11 23 25
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    return {'left_elbow_angle':left_elbow_angle, 'right_elbow_angle': right_elbow_angle, 'left_shoulder_angle':left_shoulder_angle,
    'right_shoulder_angle':right_shoulder_angle, 'left_knee_angle': left_knee_angle, 'right_knee_angle':right_knee_angle,
     'right_hip_angle':right_hip_angle, 'left_hip_angle':left_hip_angle}


def classifyPose(landmarks, angles, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
        # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    left_elbow_angle = angle['left_elbow_angle'] 
    right_elbow_angle = angle['right_elbow_angle'] 
    left_shoulder_angle = angle['left_shoulder_angle'] 
    right_shoulder_angle = angle['right_shoulder_angle']
    left_knee_angle = angle['left_knee_angle'] 
    right_knee_angle =  angle['right_knee_angle']
    right_hip_angle = angle['right_hip_angle']
    left_hip_angle = angle['left_hip_angle']
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # right hip angle landmark 23 angle : 196.12574720974953 
    # left leg knee landmark 26 angle : 200.43683078532388
    # right shoulder landmark 11 angle : 13, 11, 23: 166.87114792585106
     #----------------------------------------------------------------------------------------------------------------
    if left_hip_angle > 95 and left_hip_angle <120 or right_hip_angle > 95 and right_hip_angle <120:
        
        if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Crescent Pose' 

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------
               # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                # Specify the label of the pose that is tree pose.
                label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
                        # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
                
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0,0,255)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (90, 90),cv2.FONT_HERSHEY_PLAIN, 2, color, 5)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
        
    else:
                # Return the output image and the classified label.
        return output_image, label


def warriorPose(landmarks, angle, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
        # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    left_elbow_angle = angle['left_elbow_angle'] 
    right_elbow_angle = angle['right_elbow_angle'] 
    left_shoulder_angle = angle['left_shoulder_angle'] 
    right_shoulder_angle = angle['right_shoulder_angle']
    left_knee_angle = angle['left_knee_angle'] 
    right_knee_angle =  angle['right_knee_angle']
    right_hip_angle = angle['right_hip_angle']
    left_hip_angle = angle['left_hip_angle']

    label =''
    # if not(left_knee_angle > 165 and left_knee_angle < 195) and not(right_knee_angle > 165 and right_knee_angle < 195):
    #     label += "Bend Right Knee"

    if not(left_knee_angle > 80 and left_knee_angle < 130) and not(right_knee_angle > 80 and right_knee_angle < 130):
        label += "Bend Right Knee"
    if not(left_shoulder_angle > 80 and left_shoulder_angle < 110) and not(right_shoulder_angle > 80 and right_shoulder_angle < 110):
        label +="\n Straighten both hand posture"
    if not(left_knee_angle > 165 and left_knee_angle < 195) and not(right_knee_angle > 165 and right_knee_angle < 195):
        label += "\n Make left leg straight"

    # Check if the both arms are straight.
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------
               # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 100 and left_knee_angle < 130 or right_knee_angle > 100 and right_knee_angle < 130:

                    # Specify the label of the pose that is Warrior II pose.
                    label += 'Warrior II Pose - Well done' 
       
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0,0,255)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (90, 90),cv2.FONT_HERSHEY_PLAIN, 2, color, 5)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')
        
    else:
                # Return the output image and the classified label.
        return output_image, label
