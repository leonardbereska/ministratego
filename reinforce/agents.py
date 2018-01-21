import random
import numpy as np
import game
import pieces


class Agent:
    def __init__(self, team):
        self.team = team

    def init_setup(self, types_available):
        types_setup = self.decide_setup(types_available)
        pieces_setup = np.array([pieces.Piece(i, self.team) for i in types_setup])  # list of types to array of Pieces
        pieces_setup.resize((2, 5))
        return pieces_setup

    def decide_setup(self, types_available):  # to be overridden by subclass
        return np.random.choice(types_available, 10, replace=False)  # randomly order the available figures in a list

    def decide_move(self, board):
        raise NotImplementedError


class Random(Agent):
    def __init__(self, team):
        super(Random, self).__init__(team=team)

    def decide_move(self, board):
        actions = game.get_poss_actions(board, self.team)
        action = random.choice(actions)  # ignore state, do random action
        return action


# class Reinforce(Agent):
#     def __init__(self, team):
#         super(Reinforce, self).__init__(team=team)
#
#     def decide_move(self, board):

