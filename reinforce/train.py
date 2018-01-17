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
import env # env superclass
import models

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


def plot_scores(episode_scores):
    global N_SMOOTH
    plt.figure(2)
    plt.clf()
    scores_t = torch.FloatTensor(episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    average = [0]
    if len(scores_t) >= N_SMOOTH:
        means = scores_t.unfold(0, N_SMOOTH, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(N_SMOOTH-1), means))
        average = means.numpy()
        plt.plot(average)
    plt.title('Average Score over last {} Episodes: {}'.format(N_SMOOTH, int(average[-1]*10)/10))
    plt.pause(0.001)  # pause a bit so that plots are updated


def run_env(env, user_test):
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
                action = action[0, 0]
            _, done = env.step(action)
            env.show()
            if done and env.reward == env.reward_win:
                print("Won!")
            elif (done and env.reward == env.reward_loss) or env.score < -3:
                print("Lost")
                break


def select_action(state, p_random):
    """
    Agents action is one of four directions
    :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
    """
    sample = random.random()
    if sample > p_random:
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
    keys = ('w', 's', 'a', 'd', 'i', 'k', 'j', 'l')
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

    # optimize network
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
                print("\nEpisode {}/{}".format(i_episode, num_episodes))
                print("Score: {}".format(env.score))
                print("Randomness: {}".format(p_random))
                episode_scores.append(env.score)
                plot_scores(episode_scores)
                break


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.0001
EVAL = False  # evaluation mode: controls verbosity of output e.g. printing non-optimal moves

num_episodes = 1000
EPS_DECAY = 100
N_SMOOTH = 10

model = models.CNN()
# model.load_state_dict(torch.load('./reinforce/saved_models/find_flag_CNN.pkl'))
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())

env = env.Escape()
run_env(env, user_test=True)

# train(env=env, num_episodes=num_episodes)
# torch.save(model.state_dict(), './reinforce/saved_models/escape_CNN.pkl')

# run_env(env, user_test=False)
