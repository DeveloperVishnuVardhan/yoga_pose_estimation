"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh
Project: CS-5330 -> Spring 2023.
This file contains the main.py
"""

import torch.nn.functional as F
import torch
import torch.nn as nn


class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        # Class to create a simpler ANN model.
        super(ANNModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Class to create a complex model.
        super(ComplexModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            #nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            #nn.Dropout(0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.layer6 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
