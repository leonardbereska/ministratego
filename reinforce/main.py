import env
import agents
import pieces

test = env.MiniStratego(agents.Survivor(0), agents.RandomAgent(1))
while True:
    env.watch_game(test, 0.001)


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
#setup_agent1 = np.empty((2, 5), dtype=object)
setup_agent1 = []
for pos in ((i, j) for i in range(2) for j in range(5)):
    setup_agent0[pos] = pieces.Piece(good_setup2[pos], 0, (4-pos[0], 4-pos[1]))
for pos, type in np.ndenumerate(rd_setup):
    if not type != type:  # check if type is NaN
        setup_agent1.append(pieces.Piece(int(type), 1, pos))
agent0 = agent.OmniscientExpectiSmart(0, setup_agent0)
agent1 = agent.RandomAgent(1, setup_agent1)
env = MiniStratego(agent0, agent1)
while True:
    watch_game(env, 0.001)