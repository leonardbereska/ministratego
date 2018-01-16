"""
Agent decides the initial setup and decides which action to take
"""

import random
import numpy as np
import pieces
import copy
from collections import Counter
from scipy import spatial
from scipy import optimize
import battleMatrix


class Agent:
    def __init__(self, team):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = None
        self.board = np.empty((5, 5), dtype=object)
        self.move_count = 0

        self.last_N_moves = []
        self.pieces_last_N_Moves_beforePos = []
        self.pieces_last_N_Moves_afterPos = []

        obstacle = pieces.Piece(99, 99, (2,2))
        obstacle.hidden = False
        self.board[2, 2] = obstacle  # set obstacle

        self.battleMatrix = battleMatrix.get_battle_matrix()

        # fallen pieces bookkeeping
        self.deadPieces = []
        dead_piecesdict = dict()
        types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        for type_ in set(types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        self.ordered_opp_pieces = []
        self.chances_array = np.ones((10, 10))

    def install_opp_setup(self, opp_setup):
        # Agent 1 is always in 0:2,0:5 and agent 0 in 3:5, 0:5
        if self.team == 1:
            for pos in ((i, j) for i in range(3, 5) for j in range(5)):
                piece = opp_setup[4-pos[0], 4-pos[1]]
                piece.hidden = True
                self.ordered_opp_pieces.append(piece)
                self.board[pos] = piece
            self.board[3:5, 0:5] = opp_setup
        else:
            for pos in ((i,j) for i in range(2) for j in range(5)):
                piece = opp_setup[pos]
                piece.hidden = True
                self.ordered_opp_pieces.append(piece)
                self.board[pos] = piece
            self.board[0:2, 0:5] = opp_setup


    def init_setup(self, *args):
        """
        Agent has to implement how to order his initial figures
        """
        raise NotImplementedError

    def decide_setup(self, given_setup=None):
        """
        Communicate initial setup of the available figure types to game
        """
        if given_setup is not None:
            types_setup = self.init_setup(given_setup)

            # check if used only types available
            check_setup = copy.deepcopy(types_setup)
            check_setup.sort()
            check_setup.resize(1,10)
            check_setup = check_setup[0]
            setup_valid = check_setup == given_setup

            assert(np.all(setup_valid)), "cheated in setup!"
        else:
            if self.setup is not None:
                types_setup = self.setup
            else:
                types_setup = np.random.choice(battleMatrix.get_smart_setups())

        if self.team == 0:
            for pos, piece in np.ndenumerate(types_setup):
                own_piece = pieces.Piece(piece, self.team, (4 - pos[0], 4 - pos[1]))
                own_piece.hidden = False
                self.board[4-pos[0], 4-pos[1]] = own_piece
            self.setup = self.board[3:5,0:5]
        else:
            for pos, piece in np.ndenumerate(types_setup):
                own_piece = pieces.Piece(piece, self.team, pos)
                own_piece.hidden = False
                self.board[pos] = own_piece
            self.setup = self.board[0:2, 0:5]
        return self.setup

    def update_board(self, updated_piece, board=None):
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position = updated_piece[0]
        board[updated_piece[0]] = updated_piece[1]
        return board

    def decide_move(self):
        """
        Agent has to implement on which action to decide given the state
        """
        raise NotImplementedError

    def do_move(self, move, board=None, bookkeeping=True, true_gameplay=False):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        turn = self.move_count % 2
        fight_outcome = None
        if board is None:
            board = self.board
            board[from_].has_moved = True
        moving_piece = board[from_]
        attacked_field = board[to_]
        self.last_N_moves.append(move)
        self.pieces_last_N_Moves_afterPos.append(attacked_field)
        self.pieces_last_N_Moves_beforePos.append(moving_piece)
        if not board[to_] is None:  # Target field is not empty, then has to fight
            if board is None:
                # only uncover them when the real board is being played on
                attacked_field.hidden = False
                moving_piece.hidden = False
            fight_outcome = self.fight(moving_piece, attacked_field, collect_dead_pieces=bookkeeping)
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, moving_piece), board=board)
                self.update_board((from_, None), board=board)
            elif fight_outcome == 0:
                self.update_board((to_, None), board=board)
                self.update_board((from_, None), board=board)
            else:
                self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.team:
                    self.update_prob_by_fight(attacked_field)
                else:
                    self.update_prob_by_fight(moving_piece)
        else:
            self.update_board((to_, moving_piece), board=board)
            self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.other_team:
                    self.update_prob_by_move(moving_piece)

        return board, fight_outcome

    def is_legal_move(self, move_to_check, board):
        """
        :param move_to_check: array/tuple with the coordinates of the position from and to
        :return: True if warrants a legal move, False if not
        """
        pos_before = move_to_check[0]
        pos_after = move_to_check[1]

        if board[pos_before] is None:
            return False  # no piece on field to move
        if not board[pos_after] is None:
            if board[pos_after].team == board[pos_before].team:
                return False  # cant fight own pieces
            if board[pos_after].type == 99:
                return False  # cant fight obstacles
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
        return True

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

    def update_prob_by_fight(self, *args):
        pass

    def update_prob_by_move(self, *args):
        pass

    def get_poss_actions(self, board, player):
        '''
        :param board: Fully set playboard! This function only works after enemy pieces have been assigned before!
        :return: list of possible actions for opponent
        '''
        actions_possible = []
        for pos, piece in np.ndenumerate(board):
            if piece is not None:  # board positions has a piece on it
                if not piece.type == 99:  # that piece is not an obstacle
                    if piece.team == player:
                        # check which moves are possible
                        if piece.can_move:
                            for pos_to in ((i, j) for i in range(5) for j in range(5)):
                                move = (pos, pos_to)
                                if self.is_legal_move(move, board):
                                    actions_possible.append(move)
        return actions_possible

    def analyze_board(self):
        pass


class RandomAgent(Agent):
    """
    Agent who chooses his initial setup and actions at random
    """
    def __init__(self, team):
        super(RandomAgent, self).__init__(team=team)

    def init_setup(self, types_available):
        # randomly order the available figures in a list
        return np.random.choice(types_available, 10, replace=False)

    def decide_move(self):
        actions = self.get_poss_actions(self.board, self.team)
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

    def init_setup(self, *args):
        return self.setup

    def decide_setup(self, *args):
        if self.team == 0:
            self.board[3:5, 0:5] = self.setup
        else:
            self.board[0:2, 0:5] = self.setup
        return self.setup

    def decide_move(self):
        actions = self.get_poss_actions(self.board, self.team)
        # ignore state, do random action
        action = random.choice(actions)
        return action


class ExpectiSmart(Agent):
    def __init__(self, team, setup=None):
        super(ExpectiSmart, self).__init__(team=team)
        self.setup = setup

        self.winFightReward = 10
        self.neutralFightReward = 2
        self.winGameReward = 100

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def init_setup(self, given_setup):
        # randomly order the available figures in a list
        return np.random.choice(given_setup, 10, replace=False)

    def decide_move(self):
        return self.minimax(max_depth=4)

    def minimax(self, max_depth):
        curr_board = self.assign_pieces_by_highest_probability(copy.deepcopy(self.board))
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        # this is what the expectimax agent will think

        my_doable_actions = self.get_poss_actions(board, self.team)

        # check for end-state scenario
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = -float('inf')
        best_action = None
        for action in my_doable_actions:
            board = self.do_move(action, board=board,  bookkeeping=False, true_gameplay=False)
            fight_result = board[1]
            board = board[0]
            temp_reward = current_reward
            if fight_result is not None:
                if fight_result == 1:
                    temp_reward += self.winFightReward
                elif fight_result == 0:
                    temp_reward += self.neutralFightReward  # both pieces die
                elif fight_result == -1:
                    temp_reward += -self.winFightReward
            new_val = self.min_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val < new_val:
                val = new_val
                best_action = action
            if val >= beta:
                self.board = self.undo_last_move(board)
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
            self.board = self.undo_last_move(board)
        return val, best_action

    def min_val(self, board, current_reward, alpha, beta, depth):
        # this is what the opponent will think, the min-player

        my_doable_actions = self.get_poss_actions(board, self.other_team)
        # check for end-state scenario first
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = float('inf')  # inital value set, so min comparison later possible
        best_action = None
        for action in my_doable_actions:
            board = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            fight_result = board[1]
            board = board[0]
            temp_reward = current_reward
            if fight_result is not None:
                if fight_result == 1:
                    temp_reward += self.winFightReward
                elif fight_result == 0:
                    temp_reward += self.neutralFightReward  # both pieces die
                elif fight_result == -1:
                    temp_reward += -self.winFightReward
            new_val = self.max_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val > new_val:
                val = new_val
                best_action = action
            if val <= alpha:
                self.board = self.undo_last_move(board)
                return val, best_action
            beta = min(beta, val)
            self.board = self.undo_last_move(board)
        return val, best_action

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

        # now get all pieces of the enemy on the board
        enemy_pieces = [(pos, board[pos]) for (pos, piece) in np.ndenumerate(board)
                        if piece is not None and piece.team == self.other_team]
        # all the unknowns
        enemy_pieces_known = [piece.type for (pos, piece) in enemy_pieces if not piece.hidden]
        enemy_pieces_unknown = [(pos, piece) for (pos, piece) in enemy_pieces if piece.hidden]

        # remove all dead enemy pieces from the list of pieces that need to be assigned to the unknown on the field
        # then append the leftover pieces to pieces_left_to_assign
        for piece_type, count in overall_counter.items():
            # this many pieces of piece_type need to be asssigned
            nr_remaining = count - self.deadPieces[self.other_team][piece_type]
            if nr_remaining > 0:  # if equal 0, then all pieces of this type already dead
                pieces_left_to_assign.extend([piece_type]*nr_remaining)
        for piece in enemy_pieces_known:
            pieces_left_to_assign.remove(piece)

        # find the piece on the board with the highest likelihood of being the current piece in the loop
        for piece_type in pieces_left_to_assign:
            likeliest_current_prob = 0
            current_assignment = None
            chosen_piece = None
            for pos, enemy_piece in enemy_pieces_unknown:
                if enemy_piece.piece_probabilites[piece_type] > likeliest_current_prob:
                    likeliest_current_prob = enemy_piece.piece_probabilites[piece_type]
                    current_assignment = pos
                    chosen_piece = enemy_piece
            enemy_pieces_unknown.remove((current_assignment, chosen_piece))
            board[current_assignment] = pieces.Piece(piece_type, self.other_team)
        return board

    def update_prob_by_fight(self, enemy_piece):
        idx_of_piece = self.ordered_opp_pieces.index(enemy_piece)
        type = enemy_piece.type
        if type == 1:
            self.chances_array[1, idx_of_piece] = 1
            self.chances_array[np.arange(10) != 1, idx_of_piece] = 0
        elif type == 2:
            self.chances_array[np.arange(10) not in [2, 3, 4], idx_of_piece] = 0
        elif type == 3:
            self.chances_array[np.arange(10) not in [5, 6], idx_of_piece] = 0
        elif type == 10:
            self.chances_array[7, idx_of_piece] = 1
            self.chances_array[np.arange(10) != 7, idx_of_piece] = 0
        elif type == 11:
            self.chances_array[np.arange(10) not in [8, 9], idx_of_piece] = 0
        self.update_chances_array()

    def update_prob_by_move(self, move, moving_piece):
        idx_of_piece = self.ordered_opp_pieces.index(moving_piece)
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            if np.all(np.concatenate(self.chances_array[0:2, idx_of_piece], self.chances_array[5:10, idx_of_piece]) == 0):
                return
            moving_piece.hidden = False
            self.chances_array[0:2, idx_of_piece] = 0
            self.chances_array[5:10, idx_of_piece] = 0
        else:
            if np.all(np.concatenate(self.chances_array[0, idx_of_piece], self.chances_array[8:10, idx_of_piece]) == 0):
                return
            self.chances_array[0, idx_of_piece] = 0
            self.chances_array[8:10, idx_of_piece] = 0
        self.update_chances_array()

    def update_chances_array(self):
        # As x needs to be a vector, not a matrix, i associate the entries of the self.chances_array matrix
        # with the x as such:
        # x_0 x_10 x_20 x_30 ....
        # x_1 x_11 x_21 x_31 ....
        # x_2 x_12 x_22 x_32 ....
        # x_3 x_13 x_23 x_33 ....
        #  .   .
        #  .   .
        #  .   .

        # A_eq gives the constraints A_eq * x = b_eq. Our equality constraints are:
        # Every previous 0-entry stays 0
        # The variables in each coloumn for the same piece-type have to take the same value
        # The row- and coloumn-sums aggregate to 1
        #
        a_eq = np.empty(100)

        # Every previous 0-entry stays 0
        zero_entries = np.where(self.chances_array == 0)
        for i in range(len(zero_entries[0])):
            # coloumn index (zero_entries[1]) gives the multiple of 10 for the variable vector of length 10*10
            # in the linear prog. Since we need to turn a 10*10 matrix into a length 100 vector
            constr = np.zeros(100)
            constr[10*zero_entries[1][i] + zero_entries[0][i]] = 1
            a_eq = np.append(a_eq, constr, axis=0)
        b_eq = np.zeros(len(zero_entries[0]))

        # The variables in each coloumn for the same piece-type have to take the same value
        for j in range(10):
            # 2.1 = 2.2
            constr = np.zeros(100)
            constr[10 * j + 2] = self.chances_array[2, j]
            constr[10 * j + 3] = -self.chances_array[3, j]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 0)
            # 2.2 = 2.3
            constr = np.zeros(100)
            constr[10 * j + 3] = self.chances_array[3, j]
            constr[10 * j + 4] = -self.chances_array[4, j]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 0)
            # 3.1 = 3.2
            constr = np.zeros(100)
            constr[10 * j + 5] = self.chances_array[5, j]
            constr[10 * j + 6] = -self.chances_array[6, j]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 0)
            # 11.1 = 11.2
            constr = np.zeros(100)
            constr[10 * j + 8] = self.chances_array[8, j]
            constr[10 * j + 9] = -self.chances_array[9, j]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 0)

        # The row- and coloumn-sums aggregate to 1
        for i in range(10):
            # i-th coloumn
            constr = np.zeros(100)
            constr[(10 * i):(10 * i + 10)] = self.chances_array[0:10, i]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 1)
            # i-th row
            constr = np.zeros(100)
            constr[i::10] = self.chances_array[i, 0:10]
            a_eq = np.append(a_eq, constr, axis=0)
            b_eq = np.append(b_eq, 1)
        objective = np.ones(100)
        solution = optimize.linprog(c=objective, A_eq=a_eq, b_eq=b_eq, method='simplex')
        if solution[4] == 0:
            x = solution[0]
            x.resize((10, 10))
            x.transpose()
            self.chances_array = np.multiply(self.chances_array, x)
        else:
            print(solution[4])
        return




    def undo_last_move(self, board):
        last_move = self.last_N_moves.pop()
        if last_move is None:
            raise ValueError("No last move to undo detected!")
        board[last_move[0]] = self.pieces_last_N_Moves_beforePos.pop()
        board[last_move[1]] = self.pieces_last_N_Moves_afterPos.pop()
        return board

    def update_board(self, updated_piece, board=None):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        :param board: the playboard that should be updated
        :return: void
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        board[pos] = piece
        return board


class OmniscientExpectiSmart(ExpectiSmart):
    def __init__(self, team, setup=None):
        super(OmniscientExpectiSmart, self).__init__(team=team, setup=setup)
        self.setup = setup
        self.winFightReward = 10
        self.neutralFightReward = 5
        self.winGameReward = 1000

    def install_opp_setup(self, opp_setup):
        super().install_opp_setup(opp_setup)
        self.unhide_all()

    def unhide_all(self):
        for pos, piece in np.ndenumerate(self.board):
            if piece is not None:
                piece.hidden = False

    def init_setup(self, *args):
        return self.setup

    def decide_setup(self, *args):
        if self.team == 0:
            self.board[3:5, 0:5] = self.setup
        else:
            self.board[0:2, 0:5] = self.setup
        return self.setup

    def minimax(self, max_depth):
        chosen_action = self.max_val(self.board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action
