import game
import agent


agent0 = agent.RandomAgent(team=0)
agent1 = agent.RandomAgent(team=1)
new_game = game.Game(agent0, agent1)

new_game.board