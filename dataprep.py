import os
import glob
import mediapipe as mp
import cv2
import pandas as pd
from tqdm import tqdm
from utils import try_literal_eval
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPrep:
    def __init__(self, root_dir: str, data_path: str = None):
        """
        Cleans and transforms the dataset suitable for training model.

        ARGS:
            root_dir: Path to the root_directory of images.
            data_path: Path to the key point features dataset.
        """
        self.root_dir = root_dir
        self.data_path = data_path

    def process_image(self, image_path: str):
        """Gets the key_feature_landmarks of the given image."""
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            pose.close()
            return result.pose_landmarks

    def get_keypoints(self, root_dir: str):
        """Stores the keypoints of each image in a dictionary in form key
        value pairs."""
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
        """Prepared the cleaned data."""
        if self.data_path is None:
            key_points_dict = self.get_keypoints(self.root_dir)
            df = pd.DataFrame(key_points_dict)
            df.to_csv("data.csv")

    def transform_data(self):
        """Transforms the cleaned data into format of training."""
        df = pd.read_csv(self.data_path)
        df1 = df.drop(["Unnamed: 0", "34"], axis=1)
        df1_cleaned = df1.applymap(try_literal_eval)
        expanded_columns = [pd.DataFrame(df1_cleaned[col].tolist(), columns=[
                                         f'{col}_{i + 1}' for i in range(3)]) for col in df1_cleaned]
        expanded_df = pd.concat(expanded_columns + [df["34"]], axis=1)
        expanded_df.to_csv("Pre_processed_data/transformed.csv")


class BatchData:
    def __init__(self, tranformed_path: str):
        self.transformed_path = tranformed_path

    def prepare_data(self, X_train, X_val, y_train, y_val):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        y_train_tensor = torch.tensor(
            y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Create dataloader for training and validation sets.
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        return train_loader, val_loader

    def split_data(self):
        df = pd.read_csv(self.transformed_path)
        X = df.drop(["Unnamed: 0", "34"], axis=1)
        y = df["34"]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        # Split the data into training and validation sets using stratification.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        return X_train, X_val, y_train, y_val

    def batch_data(self):
        X_train, X_val, y_train, y_val = self.split_data()
        train_loader, val_loader = self.prepare_data(
            X_train, X_val, y_train, y_val)
        return train_loader, val_loader
