import game
import agent
import pieces
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
        # pieces_setup = np.array([pieces.Piece(i, self.team) for i in setup0])
        # pieces_setup.resize((2, 5))
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
good_setup = np.flip(good_setup, 0)

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
#good_setup2 = np.flip(good_setup2, 0)

setup_agent0 = np.empty((2, 5), dtype=object)
setup_agent1 = np.empty((2, 5), dtype=object)
for pos in ((i, j) for i in range(2) for j in range(5)):
    setup_agent0[pos] = pieces.Piece(good_setup[pos], 0, (4-pos[0], 4-pos[1]))
    setup_agent1[pos] = pieces.Piece(good_setup2[pos], 1, pos)
#setup0 = np.flip(setup_agent0, 0)
agent_0 = agent.ExpectiSmart(0, setup_agent0)
agent_1 = agent.OmniscientExpectiSmart(1, setup_agent1)
game = game.Game(agent_0, agent_1)
result = game.run_game()
print(result)

# watch_game(agent0=agent.RandomAgent(team=0), agent1=agent.RandomAgent(team=1), step_time=0.01)
# simulation()

