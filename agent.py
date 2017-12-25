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
        self.setup = None
        self.board = np.empty((5, 5), dtype=object)
        opp_setup = np.array([pieces.Piece(88, (self.team + 1) % 2)]*10, dtype=object)
        opp_setup.resize(2, 5)
        self.board[3:5, 0:5] = opp_setup
        self.deadFigures = []
        self.deadFigures.append([])
        self.deadFigures.append([])

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
        check_setup.resize(1,10)
        check_setup = check_setup[0]
        setup_valid = check_setup == types_available

        assert(np.all(setup_valid)), "cheated in setup!"

        # converting list of types to an 2x5 array of Pieces
        pieces_setup = np.array([pieces.Piece(i, self.team) for i in types_setup])
        pieces_setup.resize(2, 5)
        self.setup = pieces_setup
        self.board[0:2, 0:5] = pieces_setup
        return pieces_setup

    def updateBoard(self, updatedPiece):
        self.board[updatedPiece[0]] = updatedPiece[1]

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

    def decide_setup(self, *args):
        self.board[0:2, 0:5] = self.setup
        return self.setup

    def decide_move(self, state, actions):
        # ignore state, do random action
        action = random.choice(actions)
        return action

class ExpectiSmart(Agent):
    def __init__(self, team):
        super(ExpectiSmart, self).__init__(team=team)
        self.winFightReward = 10
        self.gainInfoReward = 5

        self.oppPiecesProbabilites = dict()
        # each entry of this dict is a list containting the probability P_k of hidden piece j being piece k, i.e.
        # oppPiecesProbabilites[3,0] = [P_0, P_1, P_2, P_3, P_10, P_11] with indices declaring k
        for pos in ((i, j) for i in range(3,5) for j in range(0,5)):
            self.oppPiecesProbabilites[id(self.board[pos])] = [0.1, 0.1, 0.3, 0.2, 0.1, 0.2]

        self.battleMatrix = dict()
        self.battleMatrix[1, 11] = -1
        self.battleMatrix[1, 1] = 0
        self.battleMatrix[1, 2] = -1
        self.battleMatrix[1, 3] = -1
        self.battleMatrix[1, 0] = 1
        self.battleMatrix[1, 10] = 1
        self.battleMatrix[2, 0] = 1
        self.battleMatrix[2, 11] = -1
        self.battleMatrix[2, 1] = 1
        self.battleMatrix[2, 2] = 0
        self.battleMatrix[2, 3] = -1
        self.battleMatrix[2, 10] = -1
        self.battleMatrix[3, 0] = 1
        self.battleMatrix[3, 11] = 1
        self.battleMatrix[3, 2] = 1
        self.battleMatrix[3, 3] = 0
        self.battleMatrix[3, 1] = 1
        self.battleMatrix[3, 10] = -1
        self.battleMatrix[10, 0] = 1
        self.battleMatrix[10, 11] = -1
        self.battleMatrix[10, 1] = 1
        self.battleMatrix[10, 2] = 1
        self.battleMatrix[10, 3] = 1
        self.battleMatrix[10, 10] = 0

    def init_setup(self, types_available):
        return self.setup

    def decide_move(self, state, actions):
        current_reward = 0
        self.expectimax(state, actions, current_reward, max_depth=4)

    def expectimax(self, state, actions, curr_rew, max_depth):
        for action in actions:
            from_ = action[0]
            to_ = action[1]
            curr_board = copy.deepcopy(self.board)
            if self.goal_test(actions):
                return curr_rew
            if self.board[to_].team == (self.team +1 % 2):
                if self.board[to_]

    def goal_test(self, actions_possible):
        if 0 in self.deadFigures[0] or 0 in self.deadFigures[1]:
            # print('flag captured')
            return True
        elif not actions_possible:
            # print('cannot move anymore')
            return True
        else:
            return False

    def do_move(self, board, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        if not self.is_legal_move(move):
            return False  # illegal move chosen
        if not self.board[to_] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(self.board[from_], self.board[to_])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, self.board[from_]), visible=True)
                self.update_board((from_, None), visible=True)
            elif fight_outcome == 0:
                self.update_board((to_, None), visible=True)
                self.update_board((from_, None), visible=True)
            else:
                self.update_board((from_, None), visible=True)
        else:
            self.update_board((to_, self.board[from_]), visible=False)
            self.update_board((from_, None), visible=False)

        return True

    def update_board(self, updated_piece, visible):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        :param visible: boolean, True if the piece is visible to the enemy team, False if hidden
        :return: void
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        if visible:
            self.agents[0].updateBoard(updated_piece)
            self.agents[1].updateBoard(updated_piece)
        else:
            if not piece is None:
                if piece.team == 0:
                    self.agents[0].updateBoard(updated_piece)
                    self.agents[1].updateBoard((pos, pieces.Piece(88, 1)))
                else:
                    self.agents[0].updateBoard((pos, pieces.Piece(88, 0)))
                    self.agents[1].updateBoard(updated_piece)
            else:
                self.agents[0].updateBoard(updated_piece)
                self.agents[1].updateBoard(updated_piece)

        self.board[pos] = piece


    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadFigures[piece_def.team].append(piece_def.type)
        elif outcome == 0:
            self.deadFigures[piece_def.team].append(piece_def.type)
            self.deadFigures[piece_att.team].append(piece_att.type)
        elif outcome == -1:
            self.deadFigures[piece_att.team].append(piece_att.type)
        return outcome