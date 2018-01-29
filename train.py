import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import agent
import env
import helpers
import models


# training of DQN network code is adjusted from Adam Paszke
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

def optimize_model():
    """
    Sample batch from memory of environment transitions and train network to fit the
    temporal difference TD(0) Q-value approximation
    """
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
    # TODO next state value should be masked by possible actions too
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
    """
    Plots simulated games in an environment for visualization
    :param env: environment to be run
    :param n_runs: how many episodes should be run
    :return: plot of each step in the environment
    """
    for i in range(n_runs):
        env.reset()
        env.show()
        done = False
        while not done:
            state = env.agents[0].board_to_state()  # for the reinforcement agent convert board to state input
            action = env.agents[0].select_action(state, 0.00)
            action = action[0, 0]  # action is unwrapped from the LongTensor
            move = env.agents[0].action_to_move(action)  # e.g. action = 1 -> move = ((0, 0), (0, 1))
            _, done, won = env.step(move)
            env.show()
            if done and won:
                print("Won!")
            elif done and not won or env.steps > 20:
                print("Lost")
                break


def train(env, num_episodes):
    """
    Trains a reinforcement agent, acting according to the agents model
    or randomly (with exponentially decaying probability p_random)
    Each transition (state, action, next_state, reward) is stored in a memory.
    Each step in the environment is followed by a learning phase:
        a batch of memories is used to optimize the network model
    :param env: training environement
    :param num_episodes: number of training episodes
    :return:
    """
    episode_scores = []  # score = total reward
    episode_won = []  # win-ratio win = 1 loss = -1
    averages = []
    for i_episode in range(num_episodes):
            env.reset()  # initialize environment
            state = env.agents[0].board_to_state()  # initialize state
            while True:
                    # act in environment
                    p_random = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
                    action = env.agents[0].select_action(state, p_random)  # random action with p_random
                    move = env.agents[0].action_to_move(action[0, 0])
                    reward_value, done, won = env.step(move)  # environment step for action
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
                    # env.agents[1].model = model  # TODO real self play?

                    if done:
                        # after each episode print stats
                        print("Episode {}/{}".format(i_episode, num_episodes))
                        print("Score: {}".format(env.score))
                        print("Won: {}".format(won))
                        print("Noise: {}".format(p_random))
                        print("Illegal: {}/{}\n".format(env.illegal_moves, env.steps))
                        episode_scores.append(env.score)
                        episode_won.append(won)
                        if VERBOSE > 0:
                            if (i_episode+1) % PLOT_FREQUENCY == 0:
                                global N_SMOOTH
                                # helpers.plot_scores(episode_scores, N_SMOOTH)  # takes run time
                                averages = helpers.plot_stats(averages, episode_won, N_SMOOTH, PLOT_FREQUENCY)  # takes run time
                        break
            if i_episode % 500 == 2:
                    if VERBOSE > 2:
                            run_env(env, 1)


# hyperparameters
PLOT_FREQUENCY = 50
BATCH_SIZE = 256  # for faster training take a smaller batch size, not too small as batchnorm will not work otherwise
GAMMA = 0.9  # already favors reaching goal faster, no need for reward_step, the lower GAMMA the faster
EPS_START = 0.3  # for unstable models take higher randomness first
EPS_END = 0
EPS_DECAY = 100
N_SMOOTH = 100  # plotting scores averaged over this number of episodes
VERBOSE = 1  # level of printed output verbosity:
                # 1: plot averaged episode scores
                # 2: also print actions taken and rewards
                # 3: every 100 episodes run_env()
                # also helpful sometimes: printing probabilities in "select_action" function of agent

num_episodes = 10000  # training for how many episodes
agent0 = agent.ThreePieces(0)
agent1 = agent.Random(1)
agent1.model = agent0.model
env = env.ThreePieces(agent0, agent1)

model = env.agents[0].model  # optimize model of agent0

optimizer = optim.RMSprop(model.parameters())
memory = helpers.ReplayMemory(1000000)

model.load_state_dict(torch.load('./saved_models/threepieces.pkl'))  # trained against Random
# train(env, num_episodes)
# torch.save(model.state_dict(), './saved_models/fourpieces.pkl')

run_env(env, 10000)


# increased memory size -> why good?: less bias in estimate, why bad?: not on-policy anymore
# increased batch size -> good, especially with batchnorm
