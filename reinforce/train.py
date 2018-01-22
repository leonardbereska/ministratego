import random
import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import helpers
import models
import env
import agent


def select_action(state, p_random):
    """
    Agents action is one of four directions
    :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
    """
    sample = random.random()
    if sample > p_random:
        # deterministic action selection
        # output = model(Variable(state, volatile=True)).data
        # # print(output.numpy())
        # action = output.max(1)[1].view(1, 1)  # choose maximum index
        # return action

        # probabilistic action selection, network outputs state-action values in (0, 1)
        state_action_values = model(Variable(state, volatile=True))
        p = list(state_action_values.data[0].numpy())
        p = [int(p_i * 1000) / 1000 for p_i in p]
        p[3] = 1 - sum(p[0:3])  # artificially make probs sum to one
        # print(p)  # print probabilities
        action = np.random.choice(np.arange(0, 4), p=p)
        action = int(action)  # normal int not numpy int
        return torch.LongTensor([[action]])
    else:
        return torch.LongTensor([[random.randint(0, 3)]])


def optimize_model():
        if len(memory) < BATCH_SIZE:
                return  # not optimizing for not enough memory
        transitions = memory.sample(BATCH_SIZE)  # sample memories batch
        batch = helpers.Transition(*zip(*transitions))  # transpose the batch

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
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[
                0]  # what would the model predict for next
        next_state_values.volatile = False  # requires_grad = False to not mess with loss
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # compute the expected Q values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # compute Huber loss

        # optimize network
        optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
        optimizer.step()


def action_to_move(action, team):
    i = int(np.floor(action / 4))  # which piece: 0-3 is first 4-7 second etc.
    piece = env.living_pieces[team][i]  # TODO connect to environment
    piece_pos = piece.position  # where is the piece
    if piece_pos is None:
        move = (None, None)  # return illegal move
        return move
    action = action % 4  # 0-3 as direction
    moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # a piece can move in four directions
    direction = moves[action]  # action: 0-3
    pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
    pos_to = tuple(pos_to)
    move = (piece_pos, pos_to)
    return move


def train(env, num_episodes):
        episode_scores = []  # score = total reward
        for i_episode in range(num_episodes):
                env.reset()  # initialize environment
                state = env.get_state()  # initialize state
                while True:
                        # act in environment
                        p_random = EPS_END + (EPS_START - EPS_END) * math.exp(
                                -1. * i_episode / EPS_DECAY)
                        action = select_action(state, p_random)  # random action with probability p_random
                        move = action_to_move(action[0, 0], team=0)
                        reward_value, done = env.step(move)  # environment step for action
                        if VERBOSE > 1:
                                print(action[0, 0] + 1, reward_value)
                        reward = torch.FloatTensor([reward_value])

                        # save transition as memory and optimize model
                        if done:  # if terminal state
                                next_state = None
                        else:
                                next_state = env.get_state()
                        memory.push(state, action, next_state, reward)  # store the transition in memory
                        state = next_state  # move to the next state
                        optimize_model()  # one step of optimization of target network

                        if done:
                                print("Episode {}/{}".format(i_episode, num_episodes))
                                print("Score: {}".format(env.score))
                                print("Noise: {}".format(p_random))
                                print("Illegal: {}/{}\n".format(env.illegal_moves, env.steps))
                                episode_scores.append(env.score)
                                if VERBOSE > 1:
                                        helpers.plot_scores(episode_scores)  # takes run time
                                break
                # if i_episode % 100 == 2:
                #         if VERBOSE > 1:
                                # run_env(env, False, 1)




BATCH_SIZE = 128  # 128
GAMMA = 0.99
EPS_START = 0.02
EPS_END = 0.001
EPS_DECAY = 100
N_SMOOTH = 10  # plotting scores averaged over this number of episodes
EVAL = False  # evaluation mode: controls verbosity of output e.g. printing non-optimal moves
VERBOSE = 1  # level of printed output verbosity

num_episodes = 10000  # training for how many episodes

env = env.FindFlag(agent.Reinforce(0), agent.RandomAgent(1))
env.Train = True

state_dim = env.get_state().shape[1]  # state has state_dim*5*5 values (board_size * depth of representation)
model = models.Finder(state_dim)

optimizer = optim.RMSprop(model.parameters())
memory = helpers.ReplayMemory(1000)

train(env, num_episodes)
