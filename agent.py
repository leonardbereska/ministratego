"""
Agent decides the initial setup and decides which action to take
"""

import random
import numpy as np
import pieces


class Agent:
    def __init__(self, team):
        self.team = team

    def decide_setup(self, types_available):
        """
        Choose an initial setup of the available figure types
        """
        # randomly order the available figures in a list
        types_setup = np.random.choice(types_available, 10, replace=False)

        # converting list of types to an 2x5 array of Pieces
        pieces_setup = [pieces.Piece(i, self.team) for i in types_setup]
        setup = np.empty((10, 1), dtype=object)
        for i, piece in enumerate(pieces_setup):
            setup[i] = piece
        setup.resize(2, 5)
        return setup

    def decide_move(self, state, actions):
        # ignore state, do random action
        action = random.choice(actions)
        return action

