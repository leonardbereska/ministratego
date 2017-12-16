"""
Agent decides the initial setup and decides which action to take
"""

import random
import numpy as np
import pieces
import copy


class Agent:
    def __init__(self, team):
        self.team = team
        self.board = np.empty((5, 5), dtype=object)
        opp_setup = np.array([pieces.Piece(88, (self.team + 1) % 2)]*10, dtype=object)
        opp_setup.resize(2, 5)
        self.board[3:5, 0:5] = opp_setup

    def init_setup(self, types_available):
        """
        Agent has to implement how to order his initial figures
        """
        raise NotImplementedError

    def decide_setup(self, types_available):
        """
        Communicate initial setup of the available figure types to game
        """
        types_setup = self.init_setup(types_available)

        # check if used only types available
        check_setup = copy.deepcopy(types_setup)
        check_setup.sort()
        setup_valid = check_setup == types_available
        assert(setup_valid.all()), "cheated in setup!"

        # converting list of types to an 2x5 array of Pieces
        pieces_setup = [pieces.Piece(i, self.team) for i in types_setup]
        setup = np.empty((10, 1), dtype=object)
        for i, piece in enumerate(pieces_setup):
            setup[i] = piece
        setup.resize(2, 5)
        self.board[0:2, 0:5] = setup
        return setup

    def updateBoard(self, updatedPieces):
        for pos, piece in updatedPieces:
            self.board[pos] = piece

    def decide_move(self, state, actions):
        """
        Agent has to implement on which action to decide given the state
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    Agent who chooses his initial setup and actions at random
    """
    def __init__(self, team):
        super(RandomAgent, self).__init__(team=team)

    def init_setup(self, types_available):
        # randomly order the available figures in a list
        return np.random.choice(types_available, 10, replace=False)

    def decide_move(self, state, actions):
        # ignore state, do random action
        action = random.choice(actions)
        return action


class SmartSetup(Agent):
    """
    RandomAgent with smart initial setup
    """
    def __init__(self, team, setup):
        super(SmartSetup, self).__init__(team=team)
        self.setup = setup

    def init_setup(self, types_available):
        # cheating Agent: Agent 0 gets reward 100 in 100 simulations
        # [11, 10, 3, 10, 11, 3, 3, 3, 3, 10]
        return self.setup

    def decide_move(self, state, actions):
        # ignore state, do random action
        action = random.choice(actions)
        return action
