import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn as nn
import tf_mnist

class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()

        # Define the convolutions

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv2_bn = nn.BatchNorm2d(16)
        self.maxP1 = nn.MaxPool2d(3)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(1, 16, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.conv4_bn = nn.BatchNorm2d(16)
        self.maxP2 = nn.MaxPool2d(3)
        self.drop2 = nn.Dropout2d(0.2)

        self.flat = nn.Flatten()
        # self.dense = nn.Linear(, 15)
        self.soft = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv2_bn(x)
        x = self.maxP1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv4_bn(x)
        x = self.maxP2(x)
        x = self.drop2(x)

        x = self.flat(x)
        print(x.size)
        x = self.soft(x)
        return x