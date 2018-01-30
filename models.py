import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Finder(nn.Module):
    """
    Agent for solving the FindFlag environment
    """
    def __init__(self, state_dim):
        super(Finder, self).__init__()
        self.feature_size = 10 * 9  # 25
        self.conv1 = nn.Conv2d(state_dim, 10, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, padding=1, kernel_size=5)
        self.lin1 = nn.Linear(self.feature_size, 16)
        self.lin2 = nn.Linear(16, 4)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = F.sigmoid(self.lin2(x))  # Q-value is between 0 and 1
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
        x = self.lin1(x)
        x = F.sigmoid(x)
        return x


class TwoPieces(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(TwoPieces, self).__init__()
        self.feature_size = 10 * 25
        # self.conv1 = nn.Conv2d(state_dim, 10, padding=1, kernel_size=3)
        # self.conv1_bn = nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(10, 10, padding=2, kernel_size=5)
        # self.conv2_bn = nn.BatchNorm2d(10)
        # self.lin1 = nn.Linear(self.feature_size, 32)
        # # self.lin1 = nn.Linear(self.feature_size, 32)
        # self.lin2 = nn.Linear(32, action_dim)

        self.conv1 = nn.Conv2d(state_dim, 10, padding=2, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        # self.conv2 = nn.Conv2d(20, 20, padding=2, kernel_size=5)
        # self.conv2_bn = nn.BatchNorm2d(20)
        self.lin1 = nn.Linear(self.feature_size, 64)
        # self.lin1 = nn.Linear(self.feature_size, 32)
        self.lin2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        # x = F.relu(self.conv2_bn(self.conv2(x)))

        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = F.sigmoid(self.lin2(x))
        return x


class ThreePieces(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(ThreePieces, self).__init__()
        self.feature_size = 20 * 25

        self.conv1 = nn.Conv2d(state_dim, 20, padding=1, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.lin1 = nn.Linear(self.feature_size, 32)
        self.lin2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x


class FourPieces(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(FourPieces, self).__init__()
        self.feature_size = 20 * 25

        self.conv1 = nn.Conv2d(state_dim, 20, padding=1, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.lin1 = nn.Linear(self.feature_size, 32)
        self.lin2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))

        x = x.view(-1, self.feature_size)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x


class FourPiecesBomb(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(FourPiecesBomb, self).__init__()
        self.feature_size = 20 * 25

        self.conv1 = nn.Conv2d(state_dim, 20, padding=1, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(20)

        self.lin1 = nn.Linear(self.feature_size, 32)
        self.lin2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))

        x = x.view(-1, self.feature_size)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x


class Stratego(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Stratego, self).__init__()
        self.feature_size = 50 * 25

        self.conv1 = nn.Conv2d(state_dim, 50, padding=2, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(50)
        # self.conv2 = nn.Conv2d(50, 20, padding=1, kernel_size=3)
        # self.conv2_bn = nn.BatchNorm2d(20)

        self.lin1 = nn.Linear(self.feature_size, 128)
        self.lin2 = nn.Linear(128, 64)

        self.lin3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.tanh(self.conv1_bn(self.conv1(x)))
        x = F.tanh(self.conv2_bn(self.conv2(x)))

        x = x.view(-1, self.feature_size)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        x = self.lin3(x)
        x = F.sigmoid(x)
        return x

