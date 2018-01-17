import game
import agent
from matplotlib import pyplot as plt
import numpy as np


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


def simulation():
    """
    :return: tested_setups: list of setup and winning percentage
    """
    types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
    num_simulations = 10
    blue_won = 0
    red_won = 0

    for simu in range(num_simulations):  # simulate games
        setup0 = np.random.choice(types_available, 10, replace=False)
        setup1 = np.random.choice(types_available, 10, replace=False)

        agent0 = agent.SmartSetup(team=0, setup=setup0)  # not smart in this case!
        agent1 = agent.OmniscientExpectiSmart(team=1, setup=setup1, opp_setup=setup0)
        new = game.Game(agent0, agent1)
        new.show()
        if simu % 1 == 0:
            print('Red won: {}, Blue won: {}, Game {}/{}'.format(blue_won, red_won, simu, num_simulations))
        for step in range(2000):  # game longer than
            game_reward = new.run_step()
            if game_reward is not None:
                if game_reward[0] == 1:  # count wins
                    blue_won += 1
                else:
                    red_won += 1
                break

    print('\nAgent 0 (Blue) wins {} out of {} games'.format(blue_won, num_simulations))


watch_game(agent0=agent.RandomAgent(team=0), agent1=agent.RandomAgent(team=1), step_time=0.01)
# simulation()
# pickle.dump(setups, open('randominit2.p', 'wb'))
