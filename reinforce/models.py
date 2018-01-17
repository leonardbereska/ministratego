import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):  # one agent
    """
    Agent Network: takes state of environement and outputs action
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_size = 10 * 25
        self.conv1 = nn.Conv2d(4, 10, stride=1, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, stride=1, padding=1, kernel_size=3)
        self.lin1 = nn.Linear(self.feature_size, 16)
        self.lin3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin3(x))
        return x


class CNN_double(nn.Module):  # two agents
    """
    Agent Network: takes state of environement and outputs action
    """
    def __init__(self):
        super(CNN_double, self).__init__()
        self.feature_size = 10 * 25
        self.conv1 = nn.Conv2d(3, 10, stride=1, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, stride=1, padding=1, kernel_size=3)
        self.lin1 = nn.Linear(self.feature_size, 32)
        # self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.softmax(x)
        return x





