import random
import numpy as np
import pieces
import copy
# from collections import Counter
from scipy import spatial
# from scipy import optimize
import helpers
from torch.autograd import Variable
import torch
import battleMatrix
import models

class Agent:
    """
    Agent decides the initial setup and decides which action to take
    """
    def __init__(self, team, setup=None):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = setup
        self.board = np.empty((5, 5), dtype=object)
        if setup is not None:
            for idx, piece in np.ndenumerate(setup):  # board is initialized in environment
                piece.hidden = False
                self.board[piece.position] = piece
        self.living_pieces = []  # to be filled by environment
        self.board_positions = [(i, j) for i in range(5) for j in range(5)]

        self.move_count = 0

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
        self.types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        for type_ in set(self.types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        self.ordered_opp_pieces = []

    def action_represent(self, actors):  # does nothing but is important for Reinforce
        return

    def install_opp_setup(self, opp_setup):
        self.assignment_dict = dict()
        enemy_types = [piece.type for idx, piece in np.ndenumerate(opp_setup)]
        for idx, piece in np.ndenumerate(opp_setup):
            piece.potential_types = copy.copy(enemy_types)
            self.ordered_opp_pieces.append(piece)
            piece.hidden = True
            self.board[piece.position] = piece

    def update_board(self, updated_piece, board=None):
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position(updated_piece[0])
        board[updated_piece[0]] = updated_piece[1]
        return board

    def decide_move(self):
        """
        Agent has to implement which action to decide on given the state
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
                return board, fight_outcome
            else:
                return board, fight_outcome, (moving_piece, attacked_field)
        else:
            self.update_board((to_, moving_piece), board=board)
            self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.other_team:
                    self.update_prob_by_move(move, moving_piece)
                return board, fight_outcome,
            else:
                return board, fight_outcome, (moving_piece, attacked_field)

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
            if piece_att.guessed or piece_def.guessed:
                outcome *= 2
            return outcome
        if piece_att.guessed or piece_def.guessed:
            outcome *= 2
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
    def __init__(self, team, setup=None):
        super(RandomAgent, self).__init__(team=team, setup=setup)

    def decide_move(self):
        actions = helpers.get_poss_moves(self.board, self.team)
        # ignore state, do random action
        if not actions:
            return (None, None)
        else:
            return random.choice(actions)



class Reinforce(Agent):
    """
    Agent approximating action-value functions with an artificial neural network
    trained with Q-learning
    """
    def __init__(self, team, setup=None):
        super(Reinforce, self).__init__(team=team, setup=setup)
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
            for action in range(len(p)):  # mask out impossible actions
                if action not in poss_actions:
                    p[action] = 0
            normed = [float(i) / sum(p) for i in p]
            action = np.random.choice(np.arange(0, action_dim), p=normed)
            action = int(action)  # normal int not numpy int
            return torch.LongTensor([[action]])
        else:
            while True:
                # select action at random, but make sure it is a possible move
                i = random.randint(0, len(poss_actions) - 1)
                random_action = poss_actions[i]
                return torch.LongTensor([[random_action]])

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
        # self.model.load_state_dict(torch.load('./saved_models/finder.pkl'))

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
        # self.model.load_state_dict(torch.load('./saved_models/mazer.pkl'))

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
        super(ExpectiSmart, self).__init__(team=team, setup=setup)

        self.kill_reward = 10
        self.neutral_fight = 2
        self.winGameReward = 100
        self.certainty_multiplier = 1.2

        self.max_depth = 1

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def decide_move(self):
        nr_dead_enemies = sum(self.deadPieces[self.other_team].values())
        if nr_dead_enemies <=1:
            self.max_depth = 2
        elif nr_dead_enemies >=3 and nr_dead_enemies <=5:
            self.max_depth = 4
        elif nr_dead_enemies > 5 and nr_dead_enemies <=10:
            self.max_depth = 6
        return self.minimax(max_depth=self.max_depth)

    def minimax(self, max_depth):
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        # this is what the expectimax agent will think

        my_doable_actions = helpers.get_poss_moves(board, self.team)
        np.random.shuffle(my_doable_actions)
        # check for end-state scenario
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = -float('inf')
        best_action = None
        for action in my_doable_actions:
            board, fight_result, (attacker, defender) = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward
            if fight_result is not None:
                if defender.type == 0:
                    temp_reward += self.winGameReward - (self.max_depth - depth)/(self.winGameReward/self.kill_reward)
                    new_val = temp_reward
                else:
                    if fight_result == 1:
                        temp_reward += self.kill_reward
                    elif fight_result == 2:
                        temp_reward += int(self.certainty_multiplier*self.kill_reward)
                    elif fight_result == 0:
                        temp_reward += self.neutral_fight  # both pieces die
                    elif fight_result == -1:
                        temp_reward += -self.kill_reward
                    elif fight_result == -2:
                        temp_reward += -int(self.certainty_multiplier * self.kill_reward)
                    new_val = self.min_val(board, temp_reward, alpha, beta, depth - 1)[0]
            else:
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
        np.random.shuffle(my_doable_actions)
        # check for end-state scenario first
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = float('inf')  # inital value set, so min comparison later possible
        best_action = None
        for action in my_doable_actions:
            board, fight_result, (attacker, defender) = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward
            if fight_result is not None:
                if defender.type == 0:
                    temp_reward -= self.winGameReward - (self.max_depth - depth)/(self.winGameReward/self.kill_reward)
                    new_val = temp_reward
                else:
                    if fight_result == 1:
                        temp_reward += -self.kill_reward
                    elif fight_result == 2:
                        temp_reward += -int(self.certainty_multiplier*self.kill_reward)
                    elif fight_result == 0:
                        temp_reward += -self.neutral_fight  # both pieces die
                    elif fight_result == -1:
                        temp_reward += self.kill_reward
                    elif fight_result == -2:
                        temp_reward += int(self.certainty_multiplier * self.kill_reward)
                    new_val = self.max_val(board, temp_reward, alpha, beta, depth - 1)[0]
            else:
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
        enemy_piece.potential_types = [enemy_piece.type]

    def update_prob_by_move(self, move, moving_piece):
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            moving_piece.hidden = False
            moving_piece.potential_types = [moving_piece.type]
        else:
            immobile_enemy_types = [idx for idx, type in enumerate(moving_piece.potential_types)
                                    if type in [0, 11]]
            moving_piece.potential_types = np.delete(moving_piece.potential_types, immobile_enemy_types)

    def draw_consistent_enemy_setup(self, board):
        # get information about enemy pieces (how many, which alive, which types, and indices in assign. array)
        enemy_pieces = copy.deepcopy(self.ordered_opp_pieces)
        enemy_pieces_alive = [piece for piece in enemy_pieces if not piece.dead]
        types_alive = [piece.type for piece in enemy_pieces_alive]

        # do the following as long as the drawn assignment is not consistent with the current knowledge about them
        consistent = False
        sample = None
        while not consistent:
            # choose len(types_alive) many pieces randomly
            sample = np.random.choice(types_alive, len(types_alive), replace=False)
            consistent = True
            for idx, piece in enumerate(enemy_pieces_alive):
                if sample[idx] not in piece.potential_types:
                    consistent = False
        for idx, piece in enumerate(enemy_pieces_alive):
            piece.guessed = not piece.hidden
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
    def __init__(self, team, setup):
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

    def update_prob_by_fight(self, enemy_piece):
        pass

    def update_prob_by_move(self, move, moving_piece):
        pass
