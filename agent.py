"""
Agent decides the initial setup and decides which action to take
"""

import random
import numpy as np
import pieces
import copy
from collections import Counter
from scipy import spatial
import battleMatrix


class Agent:
    def __init__(self, team):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = None
        self.board = np.empty((5, 5), dtype=object)
        opp_setup = np.array([pieces.unknownPiece(self.other_team) for i in range(10)], dtype=object)
        opp_setup.resize((2, 5))

        # Agent 1 is always in 0:2,0:5 and agent 0 in 3:5, 0:5
        if self.team == 1:
            self.board[3:5, 0:5] = opp_setup
        else:
            self.board[0:2, 0:5] = opp_setup
        self.board[2, 2] = pieces.Piece(99, 99)  # set obstacle
        self.deadPieces = []
        dead_piecesdict = dict()
        types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        for type_ in set(types_available):
            dead_piecesdict[type_] = 0
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
        pieces_setup.resize((2, 5))
        self.setup = pieces_setup
        if self.team == 0:
            self.board[3:5, 0:5] = pieces_setup
        else:
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
        if self.team == 0:
            self.board[3:5, 0:5] = self.setup
        else:
            self.board[0:2, 0:5] = self.setup
        return self.setup

    def decide_move(self, state, actions):
        # ignore state, do random action
        action = random.choice(actions)
        return action


class ExpectiSmart(Agent):
    def __init__(self, team, setup):
        super(ExpectiSmart, self).__init__(team=team)
        self.setup = setup

        self.winFightReward = 10
        self.neutralFightReward = 2
        self.winGameReward = 100

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def init_setup(self, types_available):
        # randomly order the available figures in a list
        return np.random.choice(types_available, 10, replace=False)

    def decide_move(self, state, actions):
        return self.minimax(actions, max_depth=4)

    def minimax(self, actions, max_depth):
        curr_board = self.assign_pieces_by_highest_probability(copy.deepcopy(self.board))
        opponent_possible_actions = self.get_opp_actions(curr_board)
        chosen_action = self.max_val(curr_board, actions, opponent_possible_actions, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, max_player_actions, min_player_actions, current_reward, alpha, beta, depth):
        goal_check = self.goal_test(max_player_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)
        val = None
        best_action = None
        for action in max_player_actions:
            board_new = copy.deepcopy(board)
            board_new = self.do_move(board_new, action, bookkeeping=False)
            fight_result = board_new[1]
            board_new = board_new[0]
            if fight_result is not None:
                if fight_result == 1:
                    current_reward += self.winFightReward
                elif fight_result == 0:
                    current_reward += self.neutralFightReward
                if fight_result == -1:
                    current_reward += - self.winFightReward
            actions_remaining = copy.deepcopy(max_player_actions)
            actions_remaining.remove(action)
            val = max(val, self.min_val(board_new,
                                        actions_remaining,
                                        min_player_actions,
                                        current_reward,
                                        alpha, beta, depth-1))[0]
            if val >= beta:
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
        return val, best_action

    def min_val(self, board, max_player_actions, min_player_actions, current_reward, alpha, beta, depth):
        goal_check = self.goal_test(min_player_actions, board)
        if goal_check or depth == 0:
            if goal_check == True: # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)
        val = None
        best_action = None
        for action in min_player_actions:
            board_new = copy.deepcopy(board)
            board_new = self.do_move(board_new, action, bookkeeping=False)
            fight_result = board_new[1]
            board_new = board_new[0]
            if fight_result is not None:
                if fight_result == 1:
                    current_reward += self.winFightReward
                elif fight_result == 0:
                    current_reward += self.neutralFightReward
                if fight_result == -1:
                    current_reward += - self.winFightReward
            actions_remaining = copy.deepcopy(min_player_actions)
            actions_remaining.remove(action)
            val = min(val, self.max_val(board_new,
                                        max_player_actions,
                                        actions_remaining,
                                        current_reward,
                                        alpha, beta, depth-1))[0]
            if val <= alpha:
                best_action = action
                return val, best_action
            beta = min(beta, val)
        return val, best_action

    def is_legal_move(self, move_to_check, board):
        """
        :param move_to_check: array/tuple with the coordinates of the position from and to
        :return: True if warrants a legal move, False if not
        """
        pos_before = move_to_check[0]
        pos_after = move_to_check[1]

        if board[pos_before] is None:
            return False  # no piece on field to move
        move_dist = spatial.distance.cityblock(pos_before, pos_after)
        if move_dist > board[pos_before].move_radius:
            return False  # move too far for selected piece
        if move_dist > 1:
            if not pos_before[0] == pos_after[0] and not pos_before[1] == pos_after[1]:
                return False  # no diagonal moves allowed
            else:
                if pos_after[0] == pos_before[0]:
                    dist_sign = int(np.sign(pos_after[1] - pos_before[1]))
                    for k in list(range(pos_before[1] + dist_sign, pos_after[1], int(dist_sign))):
                        if board[(pos_before[0], k)] is not None:
                            return False  # pieces in the way of the move
                else:
                    dist_sign = int(np.sign(pos_after[0] - pos_before[0]))
                    for k in range(pos_before[0] + dist_sign, pos_after[0], int(dist_sign)):
                        if board[(k, pos_before[1])] is not None:
                            return False  # pieces in the way of the move
        if not board[pos_after] is None:
            if board[pos_after].team == board[pos_before].team:
                return False  # cant fight own pieces
            if board[pos_after].type == 99:
                return False  # cant fight obstacles
        return True

    def get_opp_actions(self, board):
        '''
        :param board: Fully set playboard! This function only works after enemy pieces have been assigned before!
        :return: list of possible actions for opponent
        '''
        actions_possible = []
        for pos, piece in np.ndenumerate(board):
            if piece is not None:  # board positions has a piece on it
                if not piece.type == 99:  # that piece is not an obstacle
                    if piece.team == self.other_team:
                        # check which moves are possible
                        if piece.can_move:
                            for pos_to in ((i, j) for i in range(5) for j in range(5)):
                                move = (pos, pos_to)
                                if self.is_legal_move(move, board):
                                    actions_possible.append(move)
        return actions_possible

    def goal_test(self, actions_possible, board=None):
        if board is not None:
            flag_alive = [False, False]
            for pos, piece in np.ndenumerate(board):
                if piece is not None and piece.type == 0:
                    flag_alive[piece.team] = True
            if not flag_alive[self.other_team]:
                return self.winGameReward
            if not flag_alive[self.team]:
                return -self.winGameReward
        else:
            if 0 in self.deadPieces[0] or 0 in self.deadPieces[1]:
                # print('flag captured')
                return True
        if not actions_possible:
            # print('cannot move anymore')
            return True
        else:
            return False

    def assign_pieces_by_highest_probability(self, board):
        # get all enemy pieces
        pieces_left_to_assign = []
        overall_counter = Counter([0,1,2,2,2,3,3,10,11,11])

        # remove all dead enemy pieces from the list of pieces that need to be assigned to the unknown on the field
        # then append the leftover pieces to pieces_left_to_assign
        for piece_type, count in overall_counter.items():
            # this many pieces of piece_type need to be asssigned
            nr_remaining = count - self.deadPieces[self.other_team][piece_type]
            if nr_remaining > 0:  # if equal 0, then all pieces of this type already dead
                pieces_left_to_assign.extend([piece_type]*nr_remaining)
        # now get all pieces of the enemy on the board, that are unknown
        enemy_pieces = [(pos, board[pos]) for (pos, piece) in np.ndenumerate(board)
                        if piece is not None and piece.type == 88 and piece.team == self.other_team]
        # find the piece on the board with the highest likelihood of being the current piece in the loop
        for piece_type in pieces_left_to_assign:
            likeliest_current_prob = 0
            current_assignment = None
            chosen_piece = None
            for pos, enemy_piece in enemy_pieces:
                if enemy_piece.piece_probabilites[piece_type] > likeliest_current_prob:
                    likeliest_current_prob = enemy_piece.piece_probabilites[piece_type]
                    current_assignment = pos
                    chosen_piece = enemy_piece
            enemy_pieces.remove((current_assignment, chosen_piece))
            board[current_assignment] = pieces.Piece(piece_type, self.other_team)
        return board

    def update_probabilites(self):
        # TODO: implement inferential statistics analyzing movement patterns etc
        pass

    def do_move(self, board, move, bookkeeping=True):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        fight_outcome = None
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
        return board, fight_outcome

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
                self.deadPieces[piece_def.team][piece_def.type] += 1
            elif outcome == 0:
                self.deadPieces[piece_def.team][piece_def.type] += 1
                self.deadPieces[piece_att.team][piece_att.type] += 1
            elif outcome == -1:
                self.deadPieces[piece_att.team][piece_att.type] += 1
            return outcome
        return outcome
