"""
Jyothi Vishnu Vardhan Kolla.
This is the main file which performs tasks such as
explorataroy data-preparation,model-training based on command line inputs.
"""

import sys
from dataprep import DataPrep, BatchData
from models import ANNModel, ComplexModel
from utils import train_model, plot_curves, load_model, get_data_point
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2


def main(argv):
    """
    Main function which takes commands from command-line and
    performs the task.
    """
    data_prep = int(argv[1])
    train_mode = int(argv[2])
    prediction_mode = int(argv[3])
    class_names = [name for name in os.listdir(
        "dataset") if os.path.isdir(os.path.join("dataset", name))]
    results_path = "Pre_processed_data/results.csv"
    if os.path.isfile(results_path):
        plot_curves(results_path)

    if data_prep == 1:  # Prepare and transform the data.
        ob = DataPrep(
            "dataset", "Pre_processed_data/data.csv")  # Initialize the DataPrep object.
        ob.prepare_data()  # Call the prepare_data function.
        ob.transform_data()  # Call the transform_data function.

    if train_mode == 1:  # Train the model.
        ob = BatchData("Pre_processed_data/transformed.csv")
        train_loader, val_loader = ob.batch_data()
        input_size = 99
        num_classes = len(class_names)
        model = ComplexModel(input_size, 128, num_classes)
        num_epochs = 300
        results = pd.DataFrame(train_model(model, train_loader, val_loader, num_epochs, "mps"))
        results.to_csv("results.csv")
        
        #Save the model state dictionary
        model_path = "Models/model_ANNModel.pth"
        torch.save(model.state_dict(), model_path)

    if prediction_mode == 1:
        model_path = "Models/model_ANNModel.pth"
        test_images_path = "Pre_processed_data/test_images"

        loaded_model = load_model(model_path=model_path)
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        mp_drawing = mp.solutions.drawing_utils


        fig = plt.figure(figsize=(20, 10))
        plt.style.use('fivethirtyeight')
        rows, cols = 4, 4
        i = 0
        for file in os.listdir(test_images_path):
            img_path = os.path.join(test_images_path, file)
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                key_points = pose.process(image_rgb)
                if key_points.pose_landmarks is not None:
                    key_points_tensor = get_data_point(key_points.pose_landmarks)
                    prediction = loaded_model(key_points_tensor.unsqueeze(0))
                    _, class_ = prediction.max(1)

                    annotated_image = image_rgb.copy()
                    mp_drawing.draw_landmarks(annotated_image, key_points.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    fig.add_subplot(rows, cols, i + 1)
                    plt.imshow(annotated_image)
                    plt.title(class_names[class_])
                    plt.axis('off')
                    i += 1

        pose.close()
        plt.show()



if __name__ == "__main__":
    main(sys.argv)
