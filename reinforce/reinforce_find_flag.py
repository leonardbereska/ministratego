# other modules
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# own modules
import pieces
import game

plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Env:
    """
    Environment: 5x5 board, agent piece, opponents flag piece.
    Rewards: Big negative reward for impossible action (hitting wall/obstacle)
            and small negative for every action not finding the flag.
    State: Complete board
    """

    def __init__(self):
        self.board = np.empty((5, 5), dtype=object)
        # choose board positions for obstacle, agent and flag
        board_positions = [(i, j) for i in range(5) for j in range(5)]
        board_positions.remove((2, 2))  # remove obstacle position
        # board_positions.remove((4, 4))  # place flag deterministically
        # self.flag_pos = (4, 4)
        self.obs_pos = (2, 2)
        choices = np.random.choice(len(board_positions), 2, replace=False)
        positions = []
        for choice in choices:
            positions.append(board_positions[choice])
        # self.agent_pos = positions[0]
        self.agent_pos, self.flag_pos= positions
        # self.agent_previous = self.agent_pos
        self.board[self.obs_pos] = pieces.Piece(99, 99)  # place obstacle
        self.agent = pieces.Piece(3, 0)
        self.board[self.agent_pos] = self.agent  # place agent
        self.board[self.flag_pos] = pieces.Piece(0, 1)  # place opponents flag
        # self.board[self.enemy_pos] = pieces.Piece(10, 1)  # place opponent
        self.score = 0
        self.steps = 0

    def get_state(self):
        """
        Defines state input for DQN network
        """
        """small state: agent current and previous position and flag"""
        # state = (self.agent_pos + self.flag_pos)  #
        # state = torch.FloatTensor([state])
        # return state
        """full board"""
        state_dim = 4
        board_state = np.zeros((state_dim, 5, 5))  # 5x5 board with 3 channels: 0: own, 1: obstacles, 2: opponent
        for pos in ((i, j) for i in range(5) for j in range(5)):
            if self.board[pos] is not None:  # piece on this field
                if self.board[pos].team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = 1
                elif self.board[pos].team == 1:  # opponents team
                    if self.board[pos].type == 0:  # flag
                        board_state[tuple([2] + list(pos))] = 1
                    elif self.board[pos].type == 10:  # enemy
                        board_state[tuple([3] + list(pos))] = 1
                else:  # obstacle piece
                    board_state[tuple([1] + list(pos))] = 1
        board_tensor = Tensor(board_state)
        board_tensor = board_tensor.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_tensor

    def step(self, action):
        goal_test = False
        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        direction = moves[action]  # action: 0, 1, 2, 3
        go_to_pos = [sum(x) for x in zip(self.agent_pos, direction)]  # go in this direction
        go_to_pos = tuple(go_to_pos)
        reward = 0
        # step closer to flag ?
        dist_flag_prev = abs(self.flag_pos[0] - self.agent_pos[0]) + abs(self.flag_pos[1] - self.agent_pos[1])
        if go_to_pos not in [(i, j) for i in range(0, 5) for j in range(0, 5)]:
            reward += -1  # hitting the wall
        else:
            piece = self.board[go_to_pos]
            if piece is not None:
                if piece.type == 99:
                    reward += -1  # hitting obstacle
                if piece.type == 0:
                    reward += 10
                    goal_test = True
            else:
                self.board[go_to_pos] = self.agent  # move to position
                self.board[self.agent_pos] = None
                self.agent_previous = self.agent_pos
                self.agent_pos = go_to_pos
                # reward += -0.01 * self.steps  # each step more and more difficult
                reward += -0.1

        dist_flag_now = abs(self.flag_pos[0] - self.agent_pos[0]) + abs(self.flag_pos[1] - self.agent_pos[1])
        if dist_flag_prev - dist_flag_now < 0 and EVAL:
            print('Not optimal')
        self.score += reward
        self.steps += 1
        # if self.score < -100:  # agent died
        #     reward = -10
        #     goal_test = True
        return reward, goal_test

    def reset(self):
        self.__init__()

    def run(self):
        # global EVAL
        # EVAL = True
        env.show()
        done = False
        while not done:
            state = env.get_state()
            action = select_action(state, 0.001)
            _, done = env.step(action[0, 0])
            env.show()
            if self.score < -3:  # stupid agent dies
                print("Lost")
                break
        print("Won!")

    def show(self):
        fig = plt.figure(1)
        game.print_board(env.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


class DQN(nn.Module):

    def __init__(self, state_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.first = nn.Linear(self.state_size, 16)
        self.h1 = nn.Linear(16, 16)
        self.h2 = nn.Linear(16, 16)
        self.h3 = nn.Linear(16, 16)
        self.h4 = nn.Linear(16, 16)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        x = x.view(-1, self.state_size)
        x = F.relu(self.first(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        x = F.relu(self.head(x))
        x = F.softmax(x)
        return x


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.feature_size = 10 * 25
        self.conv1 = nn.Conv2d(4, 10, stride=1, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, stride=1, padding=1, kernel_size=3)
        # self.conv3 = nn.Conv2d(10, 10, stride=1, padding=1, kernel_size=3)

        self.lin1 = nn.Linear(self.feature_size, 16)
        # self.lin2 = nn.Linear(16, 16)
        self.lin3 = nn.Linear(16, 4)

    def forward(self, x):
        # x = x.view(-1, self.state_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))

        x = x.view(-1, self.feature_size)
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.softmax(x)
        return x


def select_action(state, eps_threshold):
    """
    Agents action is one of four directions
    :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
    """
    sample = random.random()
    if sample > eps_threshold:
        action = model(Variable(state, volatile=True)).data.max(1)[1].view(1, 1)  # choose maximum index
        # p = list(model(Variable(state, volatile=True)).data[0].numpy())  # probability distribution
        # p = [int(p_i * 1000) for p_i in p]
        # p = [p_i/1000 for p_i in p]
        # p[3] = 1 - sum(p[0:3])  # artificially make probs sum to one
        # print(p)
        # action = np.random.choice(np.arange(0, 4), p=p)
        # action = int(action)  # normal int not numpy int
        # return LongTensor([[action]])
        return action
    else:
        return LongTensor([[random.randint(0, 3)]])


def user_action():  # for testing the environment
    direction = input("Type direction\n")
    keys = ('w', 's', 'a', 'd')
    if direction not in keys:
        direction = input("Try typing again\n")
    return keys.index(direction)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)


    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train():
    num_episodes = 30000
    eps_threshold = 0.5
    for episode in range(num_episodes):
        print("Episode {}".format(episode))
        env.reset()
        state = env.get_state()
        print(eps_threshold)
        while True:
            # Select and perform an action
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)

            action = select_action(state, eps_threshold)
            reward_value, done = env.step(action[0, 0])
            reward = Tensor([reward_value])

            if not done:
                next_state = env.get_state()
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()

            if done:
                print(env.score)
                break


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.0001
EPS_DECAY = 100
# EVAL = False


model = CNN()
# model = DQN(state_size=4)
model.load_state_dict(torch.load('./reinforce/find_flag_CNN.pkl'))
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

env = Env()
# env.show()
# env.run()


# ## Training routine
# train()
# torch.save(model.state_dict(), './reinforce/find_flag_CNN.pkl')

EVAL = True
# while True:
#     env.reset()
#     env.run()

