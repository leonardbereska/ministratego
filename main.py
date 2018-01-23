import numpy as np
from matplotlib import pyplot as plt

import game
import pieces
import agent
import helpers
import copy


def watch_game(agent0, agent1, step_time):
    """
    Watch two agents play against each other, step_time is
    """
    new_game = game.Game(agent0, agent1)
    done = False
    while not done:
        new_game.run_step()
        new_game.show()
        done = new_game.goal_test()
        plt.pause(step_time)

    if new_game.move_count % 2 == 1:
        outcome = "Red won!"
    else:
        outcome = "Blue won!"
    print(outcome)
    plt.title(outcome)
    plt.show(block=True)  # keep plot


def simulation(setup0=None, setup1=None):
    """
    :return: tested_setups: list of setup and winning percentage
    """
    types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
    num_simulations = 100
    blue_won = 0
    red_won = 0

    for simu in range(num_simulations):  # simulate games
        # reset setup with new setup if none given
        if setup0 is not None:
            setup_agent0 = setup0
        else:
            types_draw = np.random.choice(types_available, 10, replace=False)
            pos_agent0 = [(i, j) for i in range(3, 5) for j in range(5)]
            setup_agent0 = np.empty((2, 5), dtype=object)
            for idx in range(10):
                pos = pos_agent0[idx]
                setup_agent0[(4 - pos[0], 4 - pos[1])] = pieces.Piece(types_draw[idx], 0, pos)
        if setup1 is not None:
            setup_agent1 = setup1
        else:
            types_draw = np.random.choice(types_available, 10, replace=False)
            pos_agent1 = [(i, j) for i in range(2) for j in range(5)]
            setup_agent1 = np.empty((2, 5), dtype=object)
            for idx in range(10):
                pos = pos_agent1[idx]
                setup_agent1[pos] = pieces.Piece(types_draw[idx], 1, pos)

        # restart game
        print("Game number: " + str(simu+1))
        agent0 = agent.ExpectiSmart(team=0, setup=copy.deepcopy(setup_agent0))  # not smart in this case!
        agent1 = agent.OmniscientExpectiSmart(team=1, setup=copy.deepcopy(setup_agent1))
        new = game.Game(agent0, agent1)
        if simu % 1 == 0:
            print('Red won: {}, Blue won: {}, Game {}/{}'.format(blue_won, red_won, simu, num_simulations))
        for step in range(2000):  # game longer than
            print(helpers.print_board(new.board))
            game_reward = new.run_step()
            if game_reward is not None:
                if game_reward[0] == 1:  # count wins
                    blue_won += 1
                else:
                    red_won += 1
                break

    print('\nAgent 0 (Blue) wins {} out of {} games'.format(blue_won, num_simulations))


good_setup = np.empty((2, 5), dtype=int)
good_setup[0, 0] = 3
good_setup[0, 1] = 11
good_setup[0, 2] = 0
good_setup[0, 3] = 11
good_setup[0, 4] = 1
good_setup[1, 0] = 2
good_setup[1, 1] = 2
good_setup[1, 2] = 10
good_setup[1, 3] = 2
good_setup[1, 4] = 3
#good_setup = np.flip(good_setup, 0)

good_setup2 = np.empty((2, 5), dtype=int)
good_setup2[0, 0] = 1
good_setup2[0, 1] = 11
good_setup2[0, 2] = 0
good_setup2[0, 3] = 11
good_setup2[0, 4] = 1
good_setup2[1, 0] = 1
good_setup2[1, 1] = 1
good_setup2[1, 2] = 1
good_setup2[1, 3] = 1
good_setup2[1, 4] = 1

rd_setup = np.empty((5, 5), )
rd_setup[:, :] = None
rd_setup[0, 1] = 0
rd_setup[1, 0] = 11
rd_setup[2, 3] = 10
rd_setup[2, 1] = 10
rd_setup[1, 2] = 2
rd_setup[0, 4] = 10




setup_agent0 = np.empty((2, 5), dtype=object)
setup_agent1 = np.empty((2, 5), dtype=object)
for pos, piece in np.ndenumerate(good_setup2):
    setup_agent0[pos] = pieces.Piece(piece, 0, (4-pos[0], 4-pos[1]))
    setup_agent1[pos] = pieces.Piece(piece, 1, pos)
# for pos, type in np.ndenumerate(rd_setup):
#     if not type != type:  # check if type is NaN
#         setup_agent1.append(pieces.Piece(int(type), 1, pos))
#agent_0 = agent.OmniscientExpectiSmart(0, setup_agent0)
#agent_1 = agent.OmniscientExpectiSmart(1, setup_agent1)
#simulation(setup_agent0, setup_agent1)
simulation()
#print(result)

# watch_game(agent0=agent.RandomAgent(team=0), agent1=agent.RandomAgent(team=1), step_time=0.01)
# simulation()

# import env
# import agents
#
# test = env.MiniStratego(agents.Survivor(0), agents.RandomAgent(1))
# while True:
#     env.watch_game(test, 0.001)
#
#
# # for testing the ministratego!
# good_setup2 = np.empty((2, 5), dtype=int)
# good_setup2[0, 0] = 3
# good_setup2[0, 1] = 11
# good_setup2[0, 2] = 0
# good_setup2[0, 3] = 11
# good_setup2[0, 4] = 1
# good_setup2[1, 0] = 2
# good_setup2[1, 1] = 2
# good_setup2[1, 2] = 10
# good_setup2[1, 3] = 2
# good_setup2[1, 4] = 3
#
# rd_setup = np.empty((5, 5), )
# rd_setup[:, :] = None
# rd_setup[0, 1] = 0
# rd_setup[1, 0] = 11
# rd_setup[2, 3] = 10
# rd_setup[2, 1] = 10
# rd_setup[1, 2] = 10
# rd_setup[0, 4] = 10
#
# setup_agent0 = np.empty((2, 5), dtype=object)
# #setup_agent1 = np.empty((2, 5), dtype=object)
# setup_agent1 = []
# for pos in ((i, j) for i in range(2) for j in range(5)):
#     setup_agent0[pos] = pieces.Piece(good_setup2[pos], 0, (4-pos[0], 4-pos[1]))
# for pos, type in np.ndenumerate(rd_setup):
#     if not type != type:  # check if type is NaN
#         setup_agent1.append(pieces.Piece(int(type), 1, pos))
# agent0 = agent.OmniscientExpectiSmart(0, setup_agent0)
# agent0.board
# agent1 = agent.RandomAgent(1, setup_agent1)
# env = MiniStratego(agent0, agent1)
# while True:
#     watch_game(env, 0.001)