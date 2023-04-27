"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh
Project: CS-5330 -> Spring 2023.
This file contains the main.py
"""

import ast
import torch
import matplotlib.pyplot as plt
import pandas as pd
from models import ComplexModel


def try_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        print(f"Maformed value: {value}")
        return value


def evaluate_model(model, val_loader, criterion, device):
    # Function that performs evaluations of the model.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100 * correct / total, val_loss


def train_model(model, train_loader, val_loader, num_epochs, device):
    # function that trains the model.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Move the model to specified device.
    model.to(device)
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training loop.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * correct / total
        val_accuracy, val_loss = evaluate_model(
            model, val_loader, criterion, device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Accuracy: {val_accuracy:.2f}%")

    print(results)
    return results


def plot_curves(results_path: str):
    # function that plots the loss-curves.
    results_df = pd.read_csv(results_path)
    epochs = results_df["Unnamed: 0"]
    val_loss = results_df["val_loss"]
    val_acc = results_df["val_acc"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    axes[0].plot(epochs, val_loss, c='r')
    axes[0].set_title('Epochs VS val_loss', c='r')

    axes[1].plot(epochs, val_acc, c='g')
    axes[1].set_title('Epochs VS val_acc', c='g')

    plt.show()


def load_model(model_path: str):
    # function to load the model.
    loaded_model = ComplexModel(99, 128, 107)
    loaded_model.load_state_dict(torch.load(model_path))
    return loaded_model.eval()


def get_data_point(key_points):
    # function to get key-points.
    final_datapoint = []
    for idx, landmark in enumerate(key_points.landmark):
        x, y, z = landmark.x, landmark.y, landmark.z
        final_datapoint.append(x)
        final_datapoint.append(y)
        final_datapoint.append(z)

    final_tensor = torch.tensor(final_datapoint, dtype=torch.float)
    return final_tensor
