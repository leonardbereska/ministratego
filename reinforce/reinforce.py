# other modules
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import copy as cp  # for deepcopy

# torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# own modules
import pieces
import game

# plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class Env:
    """
    Environment: 5x5 board, agent piece, opponents flag piece.
    Rewards: Big negative reward for impossible action (hitting wall/obstacle)
            and small negative for every action not finding the flag.
    State: Complete board
    """

    def __init__(self):
        self.board = np.empty((5, 5), dtype=object)
        self.board_positions = [(i, j) for i in range(5) for j in range(5)]

        positions = cp.deepcopy(self.board_positions)
        positions.remove((2, 2))  # remove obstacle position
        # positions.remove((4, 4))  # place flag deterministically
        # self.flag_pos = (4, 4)
        self.obs_pos = (2, 2)

        # randomly select positions
        n_choices = 4
        choices = np.random.choice(len(positions), n_choices, replace=False)
        chosen = []
        for choice in choices:
            chosen.append(positions[choice])
        self.agent_pos, self.flag_pos, self.enemy_pos1, self.enemy_pos2 = chosen

        self.agent = pieces.Piece(3, 0)
        self.enemy1 = pieces.Piece(10, 1)
        self.enemy2 = pieces.Piece(10, 1)

        self.board[self.obs_pos] = pieces.Piece(99, 99)  # place obstacle
        self.board[self.agent_pos] = self.agent  # place agent
        self.board[self.flag_pos] = pieces.Piece(0, 1)  # place opponents flag
        self.board[self.enemy_pos1] = self.enemy1  # place opponent
        self.board[self.enemy_pos2] = self.enemy2  # place opponent

        self.score = 0
        self.steps = 0
        self.agent_previous = self.agent_pos
        self.goal_test = False

    def reset(self):  # resetting means freshly initializing
        self.__init__()

    def get_state(self):
        """
        Get state representation for decision network
        return: full board: 5x5xstate_dim Tensor one-hot Tensor of board with own, opponents figures
        """
        state_dim = 4
        board_state = np.zeros((state_dim, 5, 5))  # 5x5 board with 3 channels: 0: own, 1: obstacles, 2: opponent
        for pos in self.board_positions:
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
        board_tensor = torch.FloatTensor(board_state)
        board_tensor = board_tensor.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_tensor

    def step(self, action):
        reward = 0
        go_to_pos = action_to_pos(action, self.agent_pos)

        if go_to_pos not in self.board_positions:
            reward += -1  # hitting the wall
        else:
            piece = self.board[go_to_pos]
            if piece is not None:
                if piece.type == 99:
                    reward += -1  # hitting obstacle
                if piece.type == 0:
                    reward += 10
                    self.goal_test = True
            else:
                self.board[go_to_pos] = self.agent  # move to position
                self.board[self.agent_pos] = None
                self.agent_previous = self.agent_pos
                self.agent_pos = go_to_pos
                # reward += -0.01 * self.steps  # each step more and more difficult
                reward += -0.1

        r1, self.enemy_pos1 = self.piece_move(self.enemy_pos1)
        r2, self.enemy_pos2 = self.piece_move(self.enemy_pos2)
        reward += r1
        reward += r2
        # dist_flag_prev = abs(self.flag_pos[0] - self.agent_previous[0]) \
        #                  + abs(self.flag_pos[1] - self.agent_previous[1])
        # dist_flag_now = abs(self.flag_pos[0] - self.agent_pos[0]) + abs(self.flag_pos[1] - self.agent_pos[1])
        # if dist_flag_prev - dist_flag_now < 0 and EVAL:  # print only in evaluation mode
        #     print('Moved away from flag!')  # move away from flag is not optimal

        self.score += reward
        self.steps += 1
        return reward, self.goal_test

    def run(self, user_test):
        global EVAL
        EVAL = True  # switch evaluation mode on
        while True:
            env.reset()
            env.show()
            done = False
            while not done:
                state = env.get_state()
                if user_test:
                    action = user_action()
                else:
                    action = select_action(state, 0.00)
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

    def piece_move(self, piece_pos):
        reward = 0
        opp_piece = self.board[piece_pos]
        opp_action = random.randint(0, 3)
        go_to_pos = action_to_pos(opp_action, piece_pos)
        if go_to_pos in self.board_positions:
            piece = self.board[go_to_pos]
            if piece is not None:
                # if piece.type == 99:
                #     pass  # hitting obstacle
                # if piece.type == 0:
                #     pass  # cannot capture own flag
                if piece.type == 3:
                    reward -= 1  # kill agent
                    self.goal_test = True  # agent died
            else:
                self.board[go_to_pos] = opp_piece  # move to position
                self.board[piece_pos] = None
                piece_pos = go_to_pos
        return reward, piece_pos


def action_to_pos(action, init_pos):
    moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    direction = moves[action]  # action: 0, 1, 2, 3
    go_to_pos = [sum(x) for x in zip(init_pos, direction)]  # go in this direction
    go_to_pos = tuple(go_to_pos)
    return go_to_pos


def plot_scores(episode_scores):
    plt.figure(2)
    plt.clf()
    scores_t = torch.FloatTensor(episode_scores)
    plt.title('Training Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    # plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    n_smooth = 10
    if len(scores_t) >= n_smooth:
        means = scores_t.unfold(0, n_smooth, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(n_smooth-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


class CNN(nn.Module):
    """
    Agent Network: takes state of environement and outputs action
    """
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
        return torch.LongTensor([[random.randint(0, 3)]])


def user_action():  # for testing the environment
    direction = input("Type direction\n")
    keys = ('w', 's', 'a', 'd')
    if direction not in keys:
        direction = input("Try typing again\n")
    return keys.index(direction)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # not optimizing for not enough memory
    transitions = memory.sample(BATCH_SIZE)   # sample memories batch
    batch = Transition(*zip(*transitions))  # transpose the batch

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))  # zero for teminal states
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]  # what would the model predict for next
    next_state_values.volatile = False  # requires_grad = False to not mess with loss
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # compute the expected Q values

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # compute Huber loss

    optimizer.zero_grad()  # optimize towards expected q-values
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(env, num_episodes):
    episode_scores = []  # score = total reward
    for i_episode in range(num_episodes):
        env.reset()  # initialize environment
        state = env.get_state()  # initialize state
        while True:
            # act in environment
            p_random = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
            action = select_action(state, p_random)   # perform random action with with probability p = p_random
            reward_value, done = env.step(action[0, 0])  # environment step for action
            reward = torch.FloatTensor([reward_value])

            # save transistion as memory and optimize model
            if done:  # if terminal state
                next_state = None
            else:
                next_state = env.get_state()
            memory.push(state, action, next_state, reward)  # store the transition in memory
            state = next_state  # move to the next state
            optimize_model()  # one step of optimization of target network

            if done:
                print("Episode {}".format(i_episode))
                print(env.score)
                print(p_random)
                episode_scores.append(env.score)
                plot_scores(episode_scores)
                break


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.0001
EPS_DECAY = 100
EVAL = False  # evaluation mode: controls verbosity of output e.g. printing non-optimal moves

memory_size = 10000  # how many transitions should be stored
num_episodes = 100

model = CNN()
# model.load_state_dict(torch.load('./reinforce/find_flag_CNN.pkl'))
memory = ReplayMemory(memory_size)
optimizer = optim.RMSprop(model.parameters())

env = Env()
# env.run(user_test=True)  # test environment as user via keyboard

train(env=env, num_episodes=num_episodes)
torch.save(model.state_dict(), './reinforce/escape_CNN.pkl')

env.run(user_test=False)
