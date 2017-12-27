"""
Agent decides the initial setup and decides which action to take
"""

import random
import numpy as np
import pieces
import copy
from collections import Counter


class Agent:
    def __init__(self, team):
        self.team = team
        self.setup = None
        self.board = np.empty((5, 5), dtype=object)
        opp_setup = np.array([pieces.unknownPiece((self.team + 1) % 2) for i in range(10)], dtype=object)
        opp_setup.resize(2, 5)
        self.board[3:5, 0:5] = opp_setup
        self.deadPieces = []
        dead_piecesdict = dict()
        for type in set(self.types_available):
            dead_piecesdict[type] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

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
        return self.minimax(actions, max_depth=4)

    def minimax(self, actions, max_depth):
        curr_board = self.assign_pieces_by_highest_probability(copy.deepcopy(self.board))
        chosen_action = self.max_val(curr_board, actions, 0, -float("inf"), float("inf"), max_depth)
        return chosen_action

    def max_val(self, board, actions, current_reward, alpha, beta, depth):
        if self.goal_test(board, actions):
            return current_reward
        val = None
        for action in actions:
            board_new = copy.deepcopy(board)
            self.do_move(board_new, action, bookkeeping=False)
            actions_remaining = copy.deepcopy(actions)
            actions_remaining.remove(action)
            val = max(val, self.min_val(board, actions_remaining, current_reward, alpha, beta, depth-1))
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_val(self, board, actions, current_reward, alpha, beta, depth):
        if self.goal_test(board):
            return current_reward
        val = None
        for action in actions:
            board_new = copy.deepcopy(board)
            board_new = self.do_move(board_new, action, bookkeeping=False)
            actions_remaining = copy.deepcopy(actions)
            actions_remaining.remove(action)
            val = max(val, self.max_val(board_new, actions_remaining, current_reward, alpha, beta, depth-1))
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val

    def goal_test(self, actions_possible, board=None):
        # TODO: Make goal test check on the specific, given board instead of overall game
        # TODO: Necessary for minimax evaluation
        if 0 in self.deadPieces[0] or 0 in self.deadPieces[1]:
            # print('flag captured')
            return True
        elif not actions_possible:
            # print('cannot move anymore')
            return True
        else:
            return False

    def assign_pieces_by_highest_probability(self, board):
        # get all enemy pieces
        pieces_left_to_assign = []
        overall_counter = Counter([0,1,2,2,2,3,3,10,11,11])
        for piece_type, count in overall_counter.items():
            nr_remaining = count - self.deadPieces[self.team][piece_type]
            if nr_remaining > 0:
                pieces_left_to_assign.extend([piece_type]*nr_remaining)
        enemy_pieces = [(pos, board[pos]) for (pos, piece) in np.ndenumerate(board)
                        if piece.type == 88 and piece.team == (self.team+1) % 2]
        for piece_type in pieces_left_to_assign:
            likeliest_current_prob = 0
            current_assignment = None
            for pos, enemy_piece in enemy_pieces:
                if enemy_piece.piece_probabilites[piece_type] > likeliest_current_prob:
                    likeliest_current_prob = enemy_piece.piece_probabilites[piece_type]
                    current_assignment = pos
            board[current_assignment] = pieces.Piece(piece_type, (self.team+1) % 2)
        return board

    def update_probabilites(self):
        # TODO: implement inferential statistics analyzing movement patterns etc


    def do_move(self, board, move, bookkeeping=True):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        if not board[to_] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(board[from_], board[to_], collect_dead_pieces=bookkeeping)
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board(board, (to_, board[from_]))
                self.update_board(board, (from_, None))
            elif fight_outcome == 0:
                self.update_board(board, (to_, None))
                self.update_board(board, (from_, None))
            else:
                self.update_board(board, (from_, None))
        else:
            self.update_board(board, (to_, board[from_]))
            self.update_board(board, (from_, None))
        return board

    def update_board(self, board, updated_piece):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        :param board: the playboard that should be updated
        :return: void
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        board[pos] = piece
        return board

    def fight(self, piece_att, piece_def, collect_dead_pieces=True):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if collect_dead_pieces:
            if outcome == 1:
                self.deadPieces[piece_def.team].append(piece_def.type)
            elif outcome == 0:
                self.deadPieces[piece_def.team].append(piece_def.type)
                self.deadPieces[piece_att.team].append(piece_att.type)
            elif outcome == -1:
                self.deadPieces[piece_att.team].append(piece_att.type)
            return outcome
        return outcome
