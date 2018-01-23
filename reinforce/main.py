import env
import agents


test = env.MiniStratego(agents.Survivor(0), agents.RandomAgent(1))
while True:
    env.watch_game(test, 0.001)
