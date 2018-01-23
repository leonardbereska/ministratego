"""
Agent decides the initial setup and decides which action to take
"""
import random
import numpy as np
import pieces
import copy
from scipy import spatial
from scipy import optimize
from torch.autograd import Variable
import torch

import battleMatrix
import helpers
import models


class Agent:
    def __init__(self, team):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = None
        self.board = np.empty((5, 5), dtype=object)
        self.move_count = 0
        self.living_pieces = []  # to be filled by environment
        self.board_positions = [(i, j) for i in range(5) for j in range(5)]

        self.last_N_moves = []
        self.pieces_last_N_Moves_beforePos = []
        self.pieces_last_N_Moves_afterPos = []

        obstacle = pieces.Piece(99, 99, (2, 2))
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
            for pos in ((i, j) for i in range(2) for j in range(5)):
                piece = opp_setup[pos]
                piece.hidden = True
                self.ordered_opp_pieces.append(piece)
                self.board[pos] = piece
            self.board[0:2, 0:5] = opp_setup

    def update_board(self, updated_piece, board=None):
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position(updated_piece[0])
        board[updated_piece[0]] = updated_piece[1]
        return board

    def decide_move(self):
        """
        Agent has to implement on which action to decide given the state
        """
        raise NotImplementedError

    def action_represent(self, actors):  # does nothing but is important for Reinforce
        return

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
                    self.update_prob_by_move(move, moving_piece)
        return board, fight_outcome

    def fight(self, piece_att, piece_def, collect_dead_pieces=True):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if collect_dead_pieces:
            if outcome == 1:
                piece_def.dead = True
                self.deadPieces[piece_def.team][piece_def.type] += 1
            elif outcome == 0:
                self.deadPieces[piece_def.team][piece_def.type] += 1
                piece_def.dead = True
                self.deadPieces[piece_att.team][piece_att.type] += 1
                piece_att.dead = True
            elif outcome == -1:
                self.deadPieces[piece_att.team][piece_att.type] += 1
                piece_att.dead = True
            return outcome
        return outcome

    def update_prob_by_fight(self, *args):
        pass

    def update_prob_by_move(self, *args):
        pass

    def get_poss_actions(self, board, team):  # TODO: change references to helper's version
        return helpers.get_poss_moves(board, team)

    def is_legal_move(self, move_to_check, board):  # TODO: change references to helper's version
        return helpers.is_legal_move(board, move_to_check)

    def analyze_board(self):
        pass


class RandomAgent(Agent):
    """
    Agent who chooses his initial setup and actions at random
    """
    def __init__(self, team):
        super(RandomAgent, self).__init__(team=team)

    def decide_move(self):
        actions = helpers.get_poss_moves(self.board, self.team)
        # ignore state, do random action
        action = random.choice(actions)
        return action


class Reinforce(Agent):
    """
    Agent approximating action-value functions with an artificial neural network
    trained with Q-learning
    """
    def __init__(self, team):
        super(Reinforce, self).__init__(team=team)
        self.state_dim = NotImplementedError
        self.action_dim = NotImplementedError
        self.model = NotImplementedError

    def decide_move(self):
        state = self.board_to_state()
        action = self.select_action(state, p_random=0.00)
        move = self.action_to_move(action[0, 0])
        return move

    def state_represent(self):
        """
        Specify the state representation as input for the network
        """
        return NotImplementedError

    def action_represent(self, actors):
        """
        Initialize pieces to be controlled by agent (self.actors) (only known and to be set by environment)
        and list of (piece number, action number)
        e.g. for two pieces with 4 actions each: ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))
        """
        self.actors = actors
        piece_action = []
        for i, a in enumerate(self.actors):
            if a.type == 2:
                piece_action += [(i, j) for j in range(16)]
            else:
                piece_action += [(i, j) for j in range(4)]
        self.piece_action = piece_action

    def select_action(self, state, p_random, action_dim):
        """
        Agents action is one of four directions
        :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
        """
        poss_actions = self.poss_actions(action_dim=action_dim)
        if not poss_actions:
            return torch.LongTensor([[random.randint(0, action_dim-1)]])
        sample = random.random()
        if sample > p_random:
            state_action_values = self.model(Variable(state, volatile=True))
            p = list(state_action_values.data[0].numpy())
            # print("net: {}".format(np.round(p, 3)))

            for action in range(len(p)):  # mask out impossible actions
                if action not in poss_actions:
                    p[action] = 0
            normed = [float(i) / sum(p) for i in p]
            # print("mask: {}".format(np.round(normed, 3)))
            action = np.random.choice(np.arange(0, action_dim), p=normed)
            action = int(action)  # normal int not numpy int
            return torch.LongTensor([[action]])
        else:
            while True:
                # select action at random, but make sure it is a possible move
                i = random.randint(0, len(poss_actions) - 1)
                random_action = poss_actions[i]
                return torch.LongTensor([[random_action]])

            # TODO : properly mask out actions if not possible
    def poss_actions(self, action_dim):
        poss_moves = helpers.get_poss_moves(self.board, 0)
        poss_actions = []
        all_actions = range(0, action_dim)
        for action in all_actions:
            move = self.action_to_move(action)
            if move in poss_moves:
                poss_actions.append(action)
        return poss_actions

    def action_to_move(self, action):
        i, action = self.piece_action[action]
        piece = self.actors[i]
        piece_pos = piece.position  # where is the piece
        if piece_pos is None:
            move = (None, None)  # return illegal move
            return move
        moves = []
        for i in range(1, 5):
            moves += [(i, 0), (-i, 0), (0, -i), (0, i)]
        direction = moves[action]  # action: 0-3
        pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
        pos_to = tuple(pos_to)
        move = (piece_pos, pos_to)
        return move


    def board_to_state(self):
        conditions = self.state_represent()
        state_dim = len(conditions)
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                for i, cond in enumerate(conditions):
                    condition, value = cond(p)
                    if condition:
                        board_state[tuple([i] + list(pos))] = value  # represent type
        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Finder(Reinforce):
    def __init__(self, team):
        super(Finder, self).__init__(team=team)
        self.action_dim = 4
        self.state_dim = len(self.state_represent())
        self.model = models.Finder(self.state_dim)
        self.model.load_state_dict(torch.load('./saved_models/finder.pkl'))

    def state_represent(self):
        own_team = lambda piece: (piece.team == 0, piece.type)
        other_flag = lambda piece: (piece.team == 1, 1)
        obstacle = lambda piece: (piece.type == 99, 1)
        return own_team, other_flag, obstacle


class Mazer(Reinforce):
    def __init__(self, team):
        super(Mazer, self).__init__(team=team)
        self.action_dim = 4
        self.state_dim = len(self.state_represent())
        self.model = models.Mazer(self.state_dim)
        self.model.load_state_dict(torch.load('./saved_models/mazer.pkl'))

    def state_represent(self):
        own_team = lambda piece: (piece.team == 0, 1)
        obstacle = lambda piece: (piece.type == 99, 1)
        other_flag = lambda piece: (piece.team == 1 and piece.type == 0, 1)
        return own_team, other_flag, obstacle


class Survivor(Reinforce):
    def __init__(self, team):
        super(Survivor, self).__init__(team=team)
        self.action_dim = 8
        self.state_dim = len(self.state_represent())
        self.model = models.Survivor(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/survivor.pkl'))

    def state_represent(self):
        own_team_three = lambda p: (p.team == 0 and p.type == 3, 1)
        own_team_ten = lambda p: (p.team == 0 and p.type == 10, 1)
        own_team_flag = lambda p: (p.team == 0 and not p.can_move, 1)
        opp_team_three = lambda p: (p.team == 0 and p.type == 3, 1)
        opp_team_ten = lambda p: (p.team == 0 and p.type == 10, 1)
        opp_team_flag = lambda p: (p.team == 0 and not p.can_move, 1)
        obstacle = lambda p: (p.type == 99, 1)
        return own_team_three, own_team_ten, own_team_flag, opp_team_three, opp_team_ten, opp_team_flag,obstacle





class ExpectiSmart(Agent):
    def __init__(self, team, setup=None):
        super(ExpectiSmart, self).__init__(team=team)
        self.setup = setup

        self.winFightReward = 10
        self.neutralFightReward = 2
        self.winGameReward = 100

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def decide_move(self):
        return self.minimax(max_depth=4)

    def minimax(self, max_depth):
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        # this is what the expectimax agent will think

        my_doable_actions = helpers.get_poss_moves(board, self.team)

        # check for end-state scenario
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement # Really?
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
                board = self.undo_last_move(board)
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
            board = self.undo_last_move(board)
        return val, best_action

    def min_val(self, board, current_reward, alpha, beta, depth):
        # this is what the opponent will think, the min-player

        my_doable_actions = helpers.get_poss_moves(board, self.other_team)
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
                board = self.undo_last_move(board)
                return val, best_action
            beta = min(beta, val)
            board = self.undo_last_move(board)
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

    def update_prob_by_fight(self, enemy_piece):
        for piece in self.ordered_opp_pieces:
            if piece.unique_identifier == enemy_piece.unique_identifier:
                equiv_piece_in_list = piece
                break
        idx_of_piece = self.ordered_opp_pieces.index(equiv_piece_in_list)
        type = enemy_piece.type
        if type == 1:
            self.chances_array[1, idx_of_piece] = 1
            self.chances_array[np.arange(10) != 1, idx_of_piece] = 0
        elif type == 2:
            self.chances_array[np.delete(np.arange(10), [2, 3, 4]), idx_of_piece] = 0
        elif type == 3:
            self.chances_array[np.delete(np.arange(10), [5, 6]), idx_of_piece] = 0
        elif type == 10:
            self.chances_array[7, idx_of_piece] = 1
            self.chances_array[np.arange(10) != 7, idx_of_piece] = 0
        elif type == 11:
            self.chances_array[np.delete(np.arange(10), [8, 9]), idx_of_piece] = 0

    def update_prob_by_move(self, move, moving_piece):
        for piece in self.ordered_opp_pieces:
            if piece.unique_identifier == moving_piece.unique_identifier:
                equiv_piece_in_list = piece
                break
        idx_of_piece = self.ordered_opp_pieces.index(equiv_piece_in_list)
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            moving_piece.hidden = False
            self.chances_array[0:2, idx_of_piece] = 0
            self.chances_array[5:10, idx_of_piece] = 0
        else:
            self.chances_array[0, idx_of_piece] = 0
            self.chances_array[8:10, idx_of_piece] = 0

    def draw_consistent_enemy_setup(self, board):
        enemy_pieces = copy.deepcopy(self.ordered_opp_pieces)
        enemy_pieces_alive = [piece for piece in enemy_pieces if not piece.dead]
        types_alive = [piece.type for piece in enemy_pieces_alive]
        indices_of_types_alive = [idx for idx, piece in enumerate(enemy_pieces) if not piece.dead]

        consistent = False
        while not consistent:
            sample = np.random.choice(types_alive, len(types_alive), replace=False)
            temp_sample = copy.copy(sample)
            ext_sample = []
            for i in range(10):
                if i in indices_of_types_alive:
                    ext_sample.append(temp_sample[0])
                    temp_sample = temp_sample[1:]
                else:
                    ext_sample.append(-1)
            sample_assignment_array = np.zeros((10, 10))
            for idx, s in enumerate(ext_sample):
                if s == 0:
                    sample_assignment_array[0, idx] = 1
                elif s == 1:
                    sample_assignment_array[1, idx] = 1
                elif s == 2:
                    sample_assignment_array[2:5, idx] = 1
                elif s == 3:
                    sample_assignment_array[5:7, idx] = 1
                elif s == 10:
                    sample_assignment_array[7, idx] = 1
                elif s == 11:
                    sample_assignment_array[8:10, idx] = 1
            check = np.multiply(sample_assignment_array, self.chances_array)
            if np.sum(check) == np.sum(sample_assignment_array):
                consistent = True
        for idx, piece in enumerate(enemy_pieces_alive):
            piece.type = sample[idx]
            if piece.type in [0, 11]:
                piece.can_move = False
                piece.move_radius = 0
            elif piece.type == 2:
                piece.can_move = True
                piece.move_radius = float('inf')
            else:
                piece.can_move = True
                piece.move_radius = 1
            piece.hidden = False
            board[piece.position] = piece
        return board

    def update_chances_array_incomplete(self):
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
            a_eq = np.vstack((a_eq, constr))
        a_eq = np.delete(a_eq, (0), axis=0)
        b_eq = np.zeros(len(zero_entries[0]))

        # The variables in each coloumn for the same piece-type have to take the same value
        for j in range(10):
            # 2.1 = 2.2
            constr = np.zeros(100)
            constr[10 * j + 2] = self.chances_array[2, j]
            constr[10 * j + 3] = -self.chances_array[3, j]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 0)
            # 2.2 = 2.3
            constr = np.zeros(100)
            constr[10 * j + 3] = self.chances_array[3, j]
            constr[10 * j + 4] = -self.chances_array[4, j]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 0)
            # 3.1 = 3.2
            constr = np.zeros(100)
            constr[10 * j + 5] = self.chances_array[5, j]
            constr[10 * j + 6] = -self.chances_array[6, j]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 0)
            # 11.1 = 11.2
            constr = np.zeros(100)
            constr[10 * j + 8] = self.chances_array[8, j]
            constr[10 * j + 9] = -self.chances_array[9, j]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 0)

        # The row- and coloumn-sums aggregate to 1
        for i in range(10):
            # i-th coloumn
            constr = np.zeros(100)
            #constr[(10 * i):(10 * i + 10)] = [1]*10
            constr[(10 * i):(10 * i + 10)] = self.chances_array[0:10, i]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 1)
            # i-th row
            constr = np.zeros(100)
            #constr[i::10] = [1]*10
            constr[i::10] = self.chances_array[i, 0:10]
            a_eq = np.vstack((a_eq, constr))
            b_eq = np.append(b_eq, 1)
        objective = np.empty(100)
        for i in range(100):
            if i % 2 ==0:
                objective[i] = 1
            else:
                objective[i] = -1
        solution = optimize.linprog(c=objective, A_eq=a_eq, b_eq=b_eq, method='simplex')
        if solution['status'] == 0:
            x = solution['x']
            x.resize(10, 10)
            x = x.transpose()
            self.chances_array = np.multiply(self.chances_array, x)
        else:
            print(solution['status'])
        return

    def undo_last_move(self, board):
        last_move = self.last_N_moves.pop()
        if last_move is None:
            raise ValueError("No last move to undo detected!")
        before_piece = self.pieces_last_N_Moves_beforePos.pop()
        board[last_move[0]] = before_piece
        # the piece at the 'before' position was the one that moved, so needs its
        # last entry in the move history deleted
        before_piece.position = last_move[0]
        #before_piece.positions_history.pop()
        board[last_move[1]] = self.pieces_last_N_Moves_afterPos.pop()
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

    def minimax(self, max_depth):
        chosen_action = self.max_val(self.board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

