import os
import glob
import mediapipe as mp
import cv2
import pandas as pd
from tqdm import tqdm
from utils import try_literal_eval

class DataPrep:
    def __init__(self, root_dir: str, data_path: str=None):
        self.root_dir = root_dir
        self.data_path = data_path

    def process_image(self, image_path: str):
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            pose.close()
            return result.pose_landmarks

    def get_keypoints(self, root_dir: str):
        keypoints_dict = {i: [] for i in range(1, 35)}

        for subdir, dirs, _ in tqdm(os.walk(root_dir)):
            for d in dirs:
                print(d)
                print(f"Completed one directory")
                image_paths = glob.glob(os.path.join(subdir, d, '*.*'))
                for image_path in image_paths:
                    print(f"completed one image")
                    pose_landmarks = self.process_image(image_path)
                    if pose_landmarks:
                        for idx, landmark in enumerate(pose_landmarks.landmark):
                            keypoints_dict[idx +
                                           1].append((landmark.x, landmark.y, landmark.z))
                        keypoints_dict[34].append(d)

        return keypoints_dict

    def prepare_data(self):
        if self.data_path is None:
            key_points_dict = self.get_keypoints(self.root_dir)
            df = pd.DataFrame(key_points_dict)
            df.to_csv("data.csv")
    
    def transform_data(self):
        df = pd.read_csv(self.data_path)
        df1 = df.drop(["Unnamed: 0", "34"], axis=1)
        df1_cleaned = df1.applymap(try_literal_eval)
        expanded_columns = [pd.DataFrame(df1_cleaned[col].tolist(), columns=[f'{col}_{i + 1}' for i in range(3)]) for col in df1_cleaned]
        expanded_df = pd.concat(expanded_columns + [df["34"]], axis=1)
        expanded_df.to_csv("Pre_processed_data/transformed.csv")


