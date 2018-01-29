import numpy as np
from matplotlib import pyplot as plt

import game
import pieces
import agent
import helpers
import copy
import env
import torch


def watch_game(agent0, agent1, step_time):
    """
    Watch two agents play against each other, step_time is
    """
    new_game = game.Game(agent0, agent1)
    done = False
    while not done:
        new_game.run_step()
        helpers.print_board(new_game.board)
        done = new_game.goal_test()
        plt.pause(step_time)

    if new_game.move_count % 2 == 1:
        outcome = "Red won!"
    else:
        outcome = "Blue won!"
    print(outcome)
    plt.title(outcome)
    plt.show(block=True)  # keep plot


def simulation(agent_type_0, agent_type_1, num_simulations, setup_0=None, setup_1=None, show_game=False):
    """
    :return: tested_setups: list of setup and winning percentage
    """
    types_available = [1, 2, 2, 2, 3, 3, 10, 11, 11]
    blue_won = 0
    blue_wins_bc_flag = 0
    blue_wins_bc_noMovesLeft = 0
    red_won = 0
    red_wins_bc_flag = 0
    red_wins_bc_noMovesLeft = 0
    rounds_counter_per_game = []
    rounds_counter_win_agent_0 = []
    rounds_counter_win_agent_1 = []

    for simu in range(num_simulations):  # simulate games
        # reset setup with new setup if none given
        if setup_0 is not None:
            setup_agent_0 = setup_0
        else:
            setup_agent_0 = np.empty((2, 5), dtype=object)
            flag_positions = [(4, j) for j in range(5)] + [(3, 2)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            flag_pos = 4 - flag_positions[flag_choice][0], 4 - flag_positions[flag_choice][1]
            setup_agent_0[flag_pos] = pieces.Piece(0, 0, flag_positions[flag_choice])

            types_draw = np.random.choice(types_available, 9, replace=False)
            positions_agent_0 = [(i, j) for i in range(3, 5) for j in range(5)]
            positions_agent_0.remove(flag_positions[flag_choice])

            for idx in range(9):
                pos = positions_agent_0[idx]
                setup_agent_0[(4 - pos[0], 4 - pos[1])] = pieces.Piece(types_draw[idx], 0, pos)
        if setup_1 is not None:
            setup_agent_1 = setup_1
        else:
            setup_agent_1 = np.empty((2, 5), dtype=object)
            flag_positions = [(0, j) for j in range(5)] + [(1, 2)]
            flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
            setup_agent_1[flag_positions[flag_choice]] = pieces.Piece(0, 1, flag_positions[flag_choice])

            types_draw = np.random.choice(types_available, 9, replace=False)
            positions_agent_1 = [(i, j) for i in range(2) for j in range(5)]
            positions_agent_1.remove(flag_positions[flag_choice])

            for idx in range(9):
                pos = positions_agent_1[idx]
                setup_agent_1[pos] = pieces.Piece(types_draw[idx], 1, pos)

        # restart game
        print("Game number: " + str(simu+1))
        assert(agent_type_0 in ["random", "expectimax", "omniscientmax", "reinforce"])
        if agent_type_0 == "random":
            agent_0 = agent.RandomAgent(team=0, setup=copy.deepcopy(setup_agent_0))
            agent_output_type_0 = "RandomAgent"
        elif agent_type_0 == "expectimax":
            agent_0 = agent.ExpectiSmart(team=0, setup=copy.deepcopy(setup_agent_0))
            agent_output_type_0 = "ExpectiAgent"
        elif agent_type_0 == "omniscientmax":
            agent_0 = agent.OmniscientExpectiSmart(team=0, setup=copy.deepcopy(setup_agent_0))
            agent_output_type_0 = "OmnniscientMinMaxAgent"
        else:
            agent_0 = agent.Reinforce(team=0, setup=copy.deepcopy(setup_agent_0))
            agent_output_type_0 = "ReinforceLearningAgent"

        assert(agent_type_1 in ["random", "expectimax", "omniscientmax", "reinforce"])
        if agent_type_1 == "random":
            agent_1 = agent.RandomAgent(team=1, setup=copy.deepcopy(setup_agent_1))
            agent_output_type_1 = "RandomAgent"
        elif agent_type_1 == "expectimax":
            agent_1 = agent.ExpectiSmart(team=1, setup=copy.deepcopy(setup_agent_1))
            agent_output_type_1 = "OmnniscientMinMaxAgent"
        elif agent_type_1 == "omniscientmax":
            agent_1 = agent.OmniscientExpectiSmart(team=1, setup=copy.deepcopy(setup_agent_1))
            agent_output_type_1 = "OmnniscientMinMaxAgent"
        else:
            agent_1 = agent.Reinforce(team=1, setup=copy.deepcopy(setup_agent_1))
            agent_output_type_1 = "ReinforceLearningAgent"
        game_ = game.Game(agent_0, agent_1)
        if simu % 1 == 0:
            print('BLUE won: {}, RED won: {}, Game {}/{}'.format(blue_won, red_won, simu, num_simulations))
            print('BLUE won by flag capture: {}, BLUE won by moves: {}, Game {}/{}'.format(blue_wins_bc_flag,
                                                                                           blue_wins_bc_noMovesLeft,
                                                                                           simu,
                                                                                           num_simulations))
            print('RED won by flag capture: {}, RED won by moves: {}, Game {}/{}'.format(red_wins_bc_flag,
                                                                                         red_wins_bc_noMovesLeft,
                                                                                         simu,
                                                                                         num_simulations))
        for step in range(2000):  # game longer than
            if show_game:
                helpers.print_board(game_.board)
            game_reward = game_.run_step()
            if game_reward is not None:
                if game_reward[0] == 1:  # count wins
                    red_won += 1
                    red_wins_bc_flag += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == 2:
                    red_won += 1
                    red_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == -1:
                    blue_won += 1
                    blue_wins_bc_flag += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                else:
                    blue_won += 1
                    blue_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                rounds_counter_per_game.append(game_.move_count)
                break
        if show_game:
            helpers.print_board(game_.board)
    file = open("{}_vs_{}_with_{}_sims.txt".format(agent_output_type_0, agent_output_type_1, num_simulations), "w")
    file.write("Statistics of {} vs. {} with {} games played.\n".format(agent_output_type_0, agent_output_type_1, num_simulations))
    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_0, red_won, num_simulations, round(100*red_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(red_wins_bc_flag, red_wins_bc_noMovesLeft))
    file.write("\nAgent {} won {}/{} games (~{}%).\n".format(agent_output_type_1, blue_won, num_simulations, round(100*blue_won/num_simulations, 2)))
    file.write("Reasons for winning: {} flag captures, {} wins through killing all enemies\n".format(blue_wins_bc_flag, blue_wins_bc_noMovesLeft))
    file.write("\nAverage game duration overall: {} rounds\n".format(round(sum(rounds_counter_per_game)/num_simulations), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_per_game)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_per_game)))
    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_0, round(sum(rounds_counter_win_agent_0)/len(rounds_counter_win_agent_0)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_0)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_0)))
    file.write("\nAverage game duration for {} wins: {} rounds\n".format(agent_output_type_1, round(sum(rounds_counter_win_agent_1)/len(rounds_counter_win_agent_1)), 2))
    file.write("Maximum number of rounds played: {} rounds\n".format(max(rounds_counter_win_agent_1)))
    file.write("Minimum number of rounds played: {} rounds\n".format(min(rounds_counter_win_agent_1)))
    file.close()


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



setup_agent0 = np.empty((2, 5), dtype=object)
setup_agent1 = np.empty((2, 5), dtype=object)
for pos, piece in np.ndenumerate(good_setup):
    setup_agent0[pos] = pieces.Piece(piece, 0, (4-pos[0], 4-pos[1]))
    setup_agent1[pos] = pieces.Piece(piece, 1, pos)
# for pos, type in np.ndenumerate(rd_setup):
#     if not type != type:  # check if type is NaN
#         setup_agent1[pos] = pieces.Piece(int(type), 1, pos)

#simulation(setup_agent0, setup_agent1)
# simulation(agent_type_0="random", agent_type_1="expectimax", num_simulations=1000)
# simulation()

def simu_env(env, n_runs=100, watch=True):
    """
    Plots simulated games in an environment for visualization
    :param env: environment to be run
    :param n_runs: how many episodes should be run
    :param watch: do you want to plot the game (watch=True) or just see the results (watch=False)?
    :return: plot of each step in the environment
    """
    n_won = 0
    n_lost = 0
    for i in range(n_runs):
        env.reset()
        env.show()
        done = False
        while not done:
            _, done, won = env.step()
            if watch:
                env.show()
            if done and won:
                n_won += 1
            elif done and not won or env.steps > 2000:  # break game that takes too long
                n_lost += 1
                break
        print("{} : {}, win ratio for Agent 0: {}".format(n_won, n_lost, n_won/(n_won+n_lost)))


# for testing the ministratego!
good_setup2 = np.empty((2, 5), dtype=int)
good_setup2[0, 0] = 3
good_setup2[0, 1] = 11
good_setup2[0, 2] = 0
good_setup2[0, 3] = 11
good_setup2[0, 4] = 1
good_setup2[1, 0] = 2
good_setup2[1, 1] = 2
good_setup2[1, 2] = 10
good_setup2[1, 3] = 2
good_setup2[1, 4] = 3

rd_setup = np.empty((5, 5), )
rd_setup[:, :] = None
rd_setup[0, 1] = 0
rd_setup[1, 0] = 11
rd_setup[2, 3] = 10
rd_setup[2, 1] = 10
rd_setup[1, 2] = 10
rd_setup[0, 4] = 10

setup_agent0 = np.empty((2, 5), dtype=object)
# setup_agent1 = np.empty((2, 5), dtype=object)

for pos in ((i, j) for i in range(2) for j in range(5)):
    setup_agent0[pos] = pieces.Piece(good_setup2[pos], 0, (4 - pos[0], 4 - pos[1]))
for pos, type in np.ndenumerate(rd_setup):
    if not type != type:  # check if type is NaN
        setup_agent1.append(pieces.Piece(int(type), 1, pos))

# board_to_setup()


simu_env(env, 1000, watch=True)

# env = env.ThreePieces(agent.ExpectiSmart(0), agent.RandomAgent(1))
# simu_env(env, 1000, watch=True)
