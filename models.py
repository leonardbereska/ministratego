import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Finder(nn.Module):
    """
    Agent for solving the Maze environment
    """
    def __init__(self, state_dim):
        super(Finder, self).__init__()
        self.feature_size = 10 * 25
        self.conv1 = nn.Conv2d(state_dim, 10, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, padding=1, kernel_size=3)
        self.lin1 = nn.Linear(self.feature_size, 16)
        self.lin2 = nn.Linear(16, 4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softmax(x)
        return x


class Mazer(nn.Module):
    """
    Agent for solving the Maze environment
    """
    def __init__(self, state_dim):
        super(Mazer, self).__init__()
        self.feature_size = 5 * 25
        self.conv1 = nn.Conv2d(state_dim, 5, stride=1, padding=1, kernel_size=3)
        self.lin0 = nn.Linear(self.feature_size, 16)
        self.lin1 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin0(x))
        x = F.relu(self.lin1(x))
        x = F.softmax(x)
        return x


class Survivor(nn.Module):
    """
    Agent for solving the Maze environment
    """
    def __init__(self, state_dim, action_dim):
        super(Survivor, self).__init__()
        self.feature_size = 100 * 25
        self.conv1 = nn.Conv2d(state_dim, 50, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, padding=1, kernel_size=3)
        self.lin1 = nn.Linear(self.feature_size, 32)

        self.lin2 = nn.Linear(32, action_dim)
        # self.lin2 = nn.Linear(16, 4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softmax(x)
        return x



