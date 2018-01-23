import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import agent
import env
import helpers
import models


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
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]  # what would the model predict
        next_state_values.volatile = False  # requires_grad = False to not mess with loss
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # compute the expected Q values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # compute Huber loss

        # optimize network
        optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
        optimizer.step()


def run_env(env, n_runs=100):
    global EVAL
    EVAL = True  # switch evaluation mode on
    for i in range(n_runs):
        env.reset()
        env.show()
        done = False
        while not done:
            # board to state
            state = env.agents[0].board_to_state()
            action = env.agents[0].select_action(state, 0.00, ACTION_DIM)
            action = action[0, 0]
            move = env.agents[0].action_to_move(action)
            _, done = env.step(move)
            env.show()
            if done and env.reward == env.reward_win:
                print("Won!")
            elif (done and env.reward == env.reward_loss) or env.score < -5:
                print("Lost")
                break


def train(env, num_episodes):
        episode_scores = []  # score = total reward
        for i_episode in range(num_episodes):
                env.reset()  # initialize environment
                state = env.agents[0].board_to_state()  # initialize state
                while True:
                        # act in environment
                        p_random = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
                        action = env.agents[0].select_action(state, p_random, ACTION_DIM)  # random action with p_random
                        move = env.agents[0].action_to_move(action[0, 0])
                        reward_value, done = env.step(move)  # environment step for action
                        if VERBOSE > 2:
                                print(action[0, 0], reward_value)
                        reward = torch.FloatTensor([reward_value])

                        # save transition as memory and optimize model
                        if done:  # if terminal state
                                next_state = None
                        else:
                                next_state = env.agents[0].board_to_state()
                        memory.push(state, action, next_state, reward)  # store the transition in memory
                        state = next_state  # move to the next state
                        optimize_model()  # one step of optimization of target network

                        if done:
                                print("Episode {}/{}".format(i_episode, num_episodes))
                                print("Score: {}".format(env.score))
                                print("Noise: {}".format(p_random))
                                print("Illegal: {}/{}\n".format(env.illegal_moves, env.steps))
                                episode_scores.append(env.score)
                                if VERBOSE > 0:
                                    global N_SMOOTH
                                    helpers.plot_scores(episode_scores, N_SMOOTH)  # takes run time
                                break
                if i_episode % 100 == 2:
                        if VERBOSE > 3:
                                run_env(env, 1)


BATCH_SIZE = 128  # 128
GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.01
EPS_DECAY = 100
N_SMOOTH = 100  # plotting scores averaged over this number of episodes
EVAL = False  # evaluation mode: controls verbosity of output e.g. printing non-optimal moves
VERBOSE = 2  # level of printed output verbosity

num_episodes = 1000  # training for how many episodes

env = env.FindFlag(agent.Finder(0), agent.RandomAgent(1))
env.Train = True

state_dim = len(env.agents[0].state_represent())  # state has state_dim*5*5 values
ACTION_DIM = 4
model = models.Finder(state_dim)

optimizer = optim.RMSprop(model.parameters())
memory = helpers.ReplayMemory(1000)


# model.load_state_dict(torch.load('./saved_models/finder.pkl'))
train(env, num_episodes)
# torch.save(model.state_dict(), './saved_models/finder.pkl')

run_env(env, 10000)
