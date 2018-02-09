import numpy as np
import game
import pieces
import agent
import env
import helpers
from timeit import default_timer as timer


def draw_random_setup(types_available, team):
    setup_agent = np.empty((2, 5), dtype=object)
    nr_pieces = len(types_available)
    if team == 0:
        flag_positions = [(4, j) for j in range(5)] + [(3, 2)]
        flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
        flag_pos = 4 - flag_positions[flag_choice][0], 4 - flag_positions[flag_choice][1]
        setup_agent[flag_pos] = pieces.Piece(0, 0, flag_positions[flag_choice])

        types_draw = np.random.choice(types_available, nr_pieces, replace=False)
        positions_agent_0 = [(i, j) for i in range(3, 5) for j in range(5)]
        positions_agent_0.remove(flag_positions[flag_choice])

        for idx in range(nr_pieces):
            pos = positions_agent_0[idx]
            setup_agent[(4 - pos[0], 4 - pos[1])] = pieces.Piece(types_draw[idx], 0, pos)
    elif team == 1:
        flag_positions = [(0, j) for j in range(5)] + [(1, 2)]
        flag_choice = np.random.choice(range(len(flag_positions)), 1)[0]
        setup_agent[flag_positions[flag_choice]] = pieces.Piece(0, 1, flag_positions[flag_choice])

        types_draw = np.random.choice(types_available, nr_pieces, replace=False)
        positions_agent_1 = [(i, j) for i in range(2) for j in range(5)]
        positions_agent_1.remove(flag_positions[flag_choice])

        for idx in range(nr_pieces):
            pos = positions_agent_1[idx]
            setup_agent[pos] = pieces.Piece(types_draw[idx], 1, pos)
    return setup_agent


def simulation(agent_type_0, agent_type_1, num_simulations, setup_0=None, setup_1=None, show_game=False):
    """
    :return: tested_setups: list of setup and winning percentage
    """
    blue_won = 0
    blue_wins_bc_flag = 0
    blue_wins_bc_noMovesLeft = 0
    red_won = 0
    red_wins_bc_flag = 0
    red_wins_bc_noMovesLeft = 0
    rounds_counter_per_game = []
    rounds_counter_win_agent_0 = []
    rounds_counter_win_agent_1 = []
    available_agents = ["random",
                        "minmax",
                        "omniscientminmax",
                        "reinforce",
                        "montecarlo",
                        "heuristic",
                        "omniscientheuristic",
                        "montecarloheuristic"]
    assert (agent_type_0 in available_agents)
    if agent_type_0 == "random":
        agent_0 = agent.Random(team=0)
        agent_output_type_0 = "RandomAgent"

    elif agent_type_0 == "heuristic":
        agent_0 = agent.Heuristic(team=0)
        agent_output_type_0 = "HeuristicAgent"

    elif agent_type_0 == "montecarloheuristic":
        agent_0 = agent.MonteCarloHeuristic(team=0)
        agent_output_type_0 = "MonteCarloHeuristicAgent"

    elif agent_type_0 == "omniscientheuristic":
        agent_0 = agent.OmniscientHeuristic(team=0)
        agent_output_type_0 = "OmniscientHeuristicAgent"

    elif agent_type_0 == "montecarlo":
        agent_0 = agent.MonteCarlo(team=0, number_of_iterations_game_sim=100)
        agent_output_type_0 = "MonteCarloAgent"

    elif agent_type_0 == "minmax":
        agent_0 = agent.MiniMax(team=0)
        agent_output_type_0 = "MinMaxAgent"

    elif agent_type_0 == "omniscientminmax":
        agent_0 = agent.Omniscient(team=0)
        agent_output_type_0 = "OmnniscientMinMaxAgent"

    else:
        agent_0 = agent.Stratego(team=0)
        agent_output_type_0 = "ReinforceLearningAgent"

    assert (agent_type_1 in available_agents)

    if agent_type_1 == "random":
        agent_1 = agent.Random(team=1)
        agent_output_type_1 = "RandomAgent"

    elif agent_type_1 == "heuristic":
        agent_1 = agent.Heuristic(team=1)
        agent_output_type_1 = "HeuristicAgent"

    elif agent_type_1 == "montecarloheuristic":
        agent_1 = agent.MonteCarloHeuristic(team=1)
        agent_output_type_1 = "MonteCarloHeuristicAgent"

    elif agent_type_1 == "omniscientheuristic":
        agent_1 = agent.OmniscientHeuristic(team=1)
        agent_output_type_1 = "OmniscientHeuristicAgent"

    elif agent_type_1 == "montecarlo":
        agent_1 = agent.MonteCarlo(team=1, number_of_iterations_game_sim=100)
        agent_output_type_1 = "MonteCarloAgent"

    elif agent_type_1 == "minmax":
        agent_1 = agent.MiniMax(team=1)
        agent_output_type_1 = "MinMaxAgent"

    elif agent_type_1 == "omniscientminmax":
        agent_1 = agent.Omniscient(team=1)
        agent_output_type_1 = "OmnniscientMinMaxAgent"

    else:
        agent_1 = agent.Stratego(team=1)
        agent_output_type_1 = "ReinforceLearningAgent"

    game_times_0 = []
    game_times_1 = []
    types = [1, 2, 2, 2, 3, 3, 10, 11, 11]
    #types = [1, 3, 10]
    for simu in range(num_simulations):  # simulate games
        # reset setup with new setup if none given
        if setup_0 is not None:
            setup_agent_0 = setup_0
        else:
            setup_agent_0 = draw_random_setup(types, 0)
        if setup_1 is not None:
            setup_agent_1 = setup_1
        else:
            setup_agent_1 = draw_random_setup(types, 1)
        agent_0.setup = setup_agent_0
        agent_1.setup = setup_agent_1
        # restart game
        game_ = game.Game(agent_0, agent_1)
        game_time_s = timer()
        if (simu+1) % 1 == 0:
            print('{} won: {}, {} won: {}, Game {}/{}'.format(agent_output_type_0,
                                                              red_won,
                                                              agent_output_type_1,
                                                              blue_won, simu,
                                                              num_simulations))
            print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_0,
                                                                                       red_wins_bc_flag,
                                                                                       agent_output_type_0,
                                                                                       red_wins_bc_noMovesLeft,
                                                                                       simu,
                                                                                       num_simulations))
            print('{} won by flag capture: {}, {} won by moves: {}, Game {}/{}'.format(agent_output_type_1,
                                                                                       blue_wins_bc_flag,
                                                                                       agent_output_type_1,
                                                                                       blue_wins_bc_noMovesLeft,
                                                                                       simu,
                                                                                       num_simulations))
        print("Game number: " + str(simu + 1))
        for step in range(2000):  # game longer than
            if show_game:
                helpers.print_board(game_.board)
            game_reward = game_.run_step()
            if game_reward is not None:
                if game_reward[0] == 1:  # count wins
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_flag += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == 2:
                    game_times_0.append(timer() - game_time_s)
                    red_won += 1
                    red_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_0.append(game_.move_count)
                elif game_reward[0] == -1:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_flag += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                else:
                    game_times_1.append(timer() - game_time_s)
                    blue_won += 1
                    blue_wins_bc_noMovesLeft += 1
                    rounds_counter_win_agent_1.append(game_.move_count)
                rounds_counter_per_game.append(game_.move_count)
                break
        if show_game:
            helpers.print_board(game_.board)
    file = open("{}_vs_{}_with_{}_sims.txt".format(agent_output_type_0, agent_output_type_1, num_simulations), "w")
    file.write("Statistics of {} vs. {} with {} games played.\n".format(agent_output_type_0, agent_output_type_1, num_simulations))
    file.write("Overall computational time of simulation: {} seconds.\n".format(sum(game_times_0) + sum(game_times_1)))

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

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_1, sum(game_times_1)/len(game_times_1)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_1)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_1)))

    file.write("\nAverage computational time for {} wins: {} seconds\n".format(agent_output_type_0, sum(game_times_0)/len(game_times_0)))
    file.write("Maximum computational time: {} seconds\n".format(max(game_times_0)))
    file.write("Minimum computational time: {} seconds\n".format(min(game_times_0)))
    file.close()
    return


# good_setups in helpers now
# good_setup = helpers.get_good_setup()
# good_setup2 = helpers.get_good_setup2()

# setup_agent0 = np.empty((2, 5), dtype=object)
# setup_agent1 = np.empty((2, 5), dtype=object)
# for pos, piece in np.ndenumerate(good_setup):
#     setup_agent0[pos] = pieces.Piece(piece, 0, (4-pos[0], 4-pos[1]))
#     setup_agent1[pos] = pieces.Piece(piece, 1, pos)
# for pos, type in np.ndenumerate(rd_setup):
#     if not type != type:  # check if type is NaN
#         setup_agent1[pos] = pieces.Piece(int(type), 1, pos)

#simulation(setup_agent0, setup_agent1)


#simulation(agent_type_0="montecarlo", agent_type_1="omniscientminmax", num_simulations=1000)

#simulation(agent_type_0="reinforce", agent_type_1="random", num_simulations=1000)


# simulation(agent_type_0="reinforce", agent_type_1="minmax", num_simulations=1000)




def simu_env(env, n_runs=1000, watch=True):
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
        # environment.show()
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
        print("{} : {}, win ratio for Agent 0: {}".format(n_won, n_lost, np.round(n_won/(n_won+n_lost), 2)))
    print("Simulation over: {} : {}, win ratio for Agent 0: {}".format(n_won, n_lost, np.round(n_won / (n_won + n_lost), 2)))


# test = env.FindFlag(agent.Finder(0), agent.Random(1))  # MinMax Heuristic
# simu_env(test, 100, watch=True)

# test = env.Stratego(agent.Stratego(0), agent.Random(1))
# simu_env(test, 100, watch=True)

# for higher depth heuristic becomes more useful somehow -> why?

# test = env.Stratego(agent.Heuristic(0, depth=2), agent.Random(1))
# test = env.Stratego(agent.MonteCarlo(0, number_of_iterations_game_sim=1000), agent.Random(1))
# Heuristic : Omniscient (depth 2) 51 : 49, win ratio for Agent 0: 0.51
# Reinforce : Random 53 : 47
# MiniMax(2) : Random 0.61 (of 100)

# Stratego : Random 0.56 (of 1000)
# Heuristic(2)(MiniMax) : Random  0.63 (of 100)
# Heuristic(2)(Omniscient) : Random  0.83 (of 100)
# Heuristic(2)(Omniscient) : Omniscient(2)  0.50 (of 100)
# Heuristic(4)(Omniscient) : Omniscient(4)  0.50 (of 100)


# simu_env(test, 100, watch=True)


environment = env.ThreePieces(agent.Heuristic(0), agent.MiniMax(1))
simu_env(environment, 100, watch=True)

# helpers.visualize_features(5000, environment, "fourpieces")

