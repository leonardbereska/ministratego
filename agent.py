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
import models


class Agent:
    """
    Agent decides which action to take
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

        self.move_count = 0  # round counter of the game
        self.last_N_moves = []
        self.pieces_last_N_Moves_beforePos = []
        self.pieces_last_N_Moves_afterPos = []

        # place obstacle on board
        obstacle = pieces.Piece(99, 99, (2, 2))
        obstacle.hidden = False
        self.board[2, 2] = obstacle

        self.battleMatrix = helpers.get_battle_matrix()

        # fallen pieces bookkeeping
        self.deadPieces = []  # TODO do we need this?
        dead_piecesdict = dict()
        self.types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        for type_ in set(self.types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        # list of enemy pieces
        self.ordered_opp_pieces = []  # TODO replace this?

    def install_board(self, board):
        """
        Install the opponents setup in this agents board and evaluate information for potential
        minmax operations later on
        :param opp_setup: numpy array of any shape with pieces objects stored in them
        :return: None
        """
        self.board = copy.deepcopy(board)
        enemy_pieces = []
        self.own_pieces = []
        for idx, piece in np.ndenumerate(self.board):
            if piece is not None:
                if piece.team == self.team:
                    self.own_pieces.append(piece)
                elif piece.team == self.other_team:
                    enemy_pieces.append(piece)
        enemy_types = [piece.type for piece in enemy_pieces]
        self.ordered_opp_pieces = np.array(enemy_pieces)
        self.own_pieces = np.array(self.own_pieces)

        for idx, piece in np.ndenumerate(self.ordered_opp_pieces):
            piece.potential_types = copy.copy(enemy_types)
            piece.hidden = True
        return None

    def action_represent(self, actors):  # does nothing but is important for Reinforce TODO deprecate
        return

    def update_board(self, updated_piece, board=None):
        """
        update the current board at the position with the piece both given by updated_piece
        :param updated_piece: tuple of (position, piece) with position being a tuple and piece being
        of class piece
        :param board: numpy array of shape (5,5) representing the board that should be updated.
        Default is the agent's own board
        :return: Updates are made in-place. No return value specified
        """
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position(updated_piece[0])
        board[updated_piece[0]] = updated_piece[1]

    def decide_move(self):
        """
        Implementation of the agent's move for the current round
        :return: tuple of "from" position tuple to "to" position tuple representing the move
        """
        raise NotImplementedError

    def do_move(self, move, board=None, bookkeeping=True, true_gameplay=False):
        """
        Execute the given move on the given board. If bookkeeping is TRUE, any dying pieces will be
        collected by the agent and stored for evaluation. If true_gameplay is TRUE, any moves made are
        due to the real game having proceeded, if FALSE it is considered a move made during planning (e.g
        minmax calculations). Return the new board and the outcome of any occuring fights during the move.
        :param move: tuple of positions tuple "from", "to" in the range of (0..4, 0..4)
        :param board: numpy array (5, 5) holding pieces objects
        :param bookkeeping: boolean
        :param true_gameplay: boolean
        :return: tuple of (numpy array (5, 5), integer)
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
        return board, fight_outcome,

    def fight(self, piece_att, piece_def, collect_dead_pieces=True):
        """
        Determine the outcome of a fight between the provided attacking piece and the defening piece.
        If collect_dead_pieces is TRUE, we will store the fallen pieces in the dict.
        Return the outcome of the fight, categorized by an integer (1=ATTACKER won, 0=TIE, -1=DEFENDER won),
        double the outcome if any of the pieces' types were not known before, but guessed
        (done for minmax calculation).
        :param piece_att: object of class Piece
        :param piece_def: object of class Piece
        :param collect_dead_pieces: boolean
        :return: integer
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

    def update_prob_by_fight(self, *args):  # TODO can we use this only in ExpectiSmart?
        pass

    def update_prob_by_move(self, *args):
        pass

    # def get_poss_actions(self, board, team):  # TODO deprecate this, all reference should go to helpers.py
    #     """
    #     get the possible actions for the agent of team "team" for the provided board.
    #     :param board: numpy array (5, 5)
    #     :param team: int (boolean 0,1)
    #     :return: list of moves
    #     """
    #     return helpers.get_poss_moves(board, team)
    #
    # def is_legal_move(self, move_to_check, board):  # TODO deprecate this
    #     """
    #     Check if the given move on the provided board is legal in terms of the rules of the game. Return
    #     a boolean TRUE for legal, FALSE for illegal.
    #     :param move_to_check: tuple of board positions
    #     :param board: numpy array (5, 5)
    #     :return: boolean
    #     """
    #     return helpers.is_legal_move(board, move_to_check)


class Random(Agent):
    """
    Agent who chooses his actions at random
    """
    def __init__(self, team, setup=None):
        super(Random, self).__init__(team=team, setup=setup)

    def decide_move(self):
        actions = helpers.get_poss_moves(self.board, self.team)
        if not actions:
            return None
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

    def select_action(self, state, p_random):
        """
        Agents action is one of four directions
        :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
        """
        poss_actions = self.poss_actions(action_dim=self.action_dim)
        if not poss_actions:
            return torch.LongTensor([[random.randint(0, self.action_dim-1)]])
        sample = random.random()
        if sample > p_random:
            state_action_values = self.model(Variable(state, volatile=True))
            q_values = list(state_action_values.data[0].numpy())

            for action in range(len(q_values)):
                if action not in poss_actions:
                    q_values[action] = -1  # (optional) mask out impossible actions
            # print("masked: {}".format(np.round(q_values, 2)))

            # 1. deterministic action selection (always select maximum q-value
            action = q_values.index(max(q_values))

            # 2. probabilistic: interpret q-value as probability, re-normalize to probabilities
            # normed = [float(i) / sum(p) for i in q_values]
            # action = np.random.choice(np.arange(0, self.action_dim), p=normed)
            # print("normed: {}".format(np.round(normed, 2)))
            # action = int(action)  # normal int not numpy int

            return torch.LongTensor([[action]])
        else:
            # 1. random from possible (not-illegal) actions
            i = random.randint(0, len(poss_actions) - 1)
            random_action = poss_actions[i]
            return torch.LongTensor([[random_action]])
            # 2. truly random (including illegal moves)
            # return torch.LongTensor([[random.randint(0, action_dim - 1)]])

    def poss_actions(self, action_dim):
        """
        Converting set of possible moves in the whole game to a set of actions for the agent
        :param action_dim: how many actions are possible for agent
        :return: list of legal actions
        """
        poss_moves = helpers.get_poss_moves(self.board, self.team)  # which moves are possible in the game
        poss_actions = []
        all_actions = range(0, action_dim)
        for action in all_actions:
            move = self.action_to_move(action)  # converting all actions to moves (which can be illegal)
            if move in poss_moves:              # only select legal moves among them
                poss_actions.append(action)
        return poss_actions

    def action_to_move(self, action):
        """
        Converting an action (integer between 0 and action_dim) to a move on the board,
        according to the action representation specified in self.piece_action
        :param action: action integer e.g. 3
        :return: move e.g. ((0, 0), (0, 1))
        """
        i, action = self.piece_action[action]
        piece = self.actors[i]
        piece_pos = piece.position  # where is the piece
        if piece_pos is None:
            move = (None, None)  # return illegal move
            return move
        moves = []
        if self.team == 0:
            for i in range(1, 5):
                moves += [(i, 0), (-i, 0), (0, -i), (0, i)]
        else:
            for i in range(1, 5):
                moves += [(-i, 0), (i, 0), (0, i), (0, -i)]  # directions reversed

        direction = moves[action]  # action: 0-3
        pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
        pos_to = tuple(pos_to)
        move = (piece_pos, pos_to)
        return move

    def board_to_state(self):
        """
        Converts the board of pieces (aka self.board) to the state input for a neural network,
        according to the environment for which the agent is specified
        (e.g. Finder only needs his own position, but MiniStratego will need all opponents pieces also)
        :return: (state_dim * 5 * 5) Tensor
        """
        conditions = self.state_represent()
        state_dim = len(conditions)
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos, val in np.ndenumerate(self.board):
            p = self.board[pos]
            # for reinforce as team 1, reverse board to have same state representation
            if self.team == 1:
                pos = (4 - pos[0], 4 - pos[1])
            if p is not None:  # piece on this field
                for i, cond in enumerate(conditions):
                    condition, value = cond(p)
                    if condition:
                        board_state[tuple([i] + list(pos))] = value  # represent type
        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Finder(Reinforce):
    """
    Agent for FindFlag environment
    """
    def __init__(self, team):
        super(Finder, self).__init__(team=team)
        self.action_dim = 4
        self.state_dim = len(self.state_represent())
        self.model = models.Finder(self.state_dim)

    def state_represent(self):
        own_team = lambda piece: (piece.team == 0, 1)
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


class TwoPieces(Reinforce):
    def __init__(self, team):
        super(TwoPieces, self).__init__(team=team)
        self.action_dim = 8
        self.state_dim = len(self.state_represent())
        self.model = models.TwoPieces(self.state_dim, self.action_dim)
        self.model.load_state_dict(torch.load('./saved_models/twopieces.pkl'))

    def state_represent(self):
        own_team_one = lambda p: (p.team == self.team and p.type == 1, 1)
        own_team_ten = lambda p: (p.team == self.team and p.type == 10, 1)
        own_team_flag = lambda p: (p.team == self.team and not p.can_move, 1)
        opp_team_one = lambda p: (p.team == self.other_team and p.type == 1, 1)
        opp_team_ten = lambda p: (p.team == self.other_team and p.type == 10, 1)
        opp_team_flag = lambda p: (p.team == self.other_team and not p.can_move, 1)
        obstacle = lambda p: (p.type == 99, 1)
        return own_team_one, own_team_ten, own_team_flag, \
               opp_team_one, opp_team_ten, opp_team_flag, obstacle


class ThreePieces(Reinforce):
    def __init__(self, team):
        super(ThreePieces, self).__init__(team=team)
        self.action_dim = 12  #
        self.state_dim = len(self.state_represent())
        self.model = models.ThreePieces(self.state_dim, self.action_dim)
        self.model.load_state_dict(torch.load('./saved_models/threepieces.pkl'))

    def state_represent(self):
        own_team_one = lambda p: (p.team == self.team and p.type == 1, 1)
        own_team_three = lambda p: (p.team == self.team and p.type == 3, 1)
        own_team_ten = lambda p: (p.team == self.team and p.type == 10, 1)
        # own_team = lambda p: (p.team == self.team and p.can_move, p.type)
        own_team_flag = lambda p: (p.team == self.team and not p.can_move, 1)
        opp_team_one = lambda p: (p.team == self.other_team and p.type == 1, 1)
        opp_team_three = lambda p: (p.team == self.other_team and p.type == 3, 1)
        opp_team_ten = lambda p: (p.team == self.other_team and p.type == 10, 1)
        opp_team_flag = lambda p: (p.team == self.other_team and not p.can_move, 1)
        obstacle = lambda p: (p.type == 99, 1)
        return own_team_one, own_team_three, own_team_ten, own_team_flag, \
               opp_team_one, opp_team_three, opp_team_ten, opp_team_flag, obstacle


class FourPieces(Reinforce):
    def __init__(self, team):
        super(FourPieces, self).__init__(team=team)
        self.action_dim = 28  # 16 (for piece 2) + 3 * 4 (for pieces 1, 3, 10)
        self.state_dim = len(self.state_represent())
        self.model = models.FourPieces(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/fourpieces.pkl'))

    def state_represent(self):
        own_team_one = lambda p: (p.team == self.team and p.type == 1, 1)
        own_team_two = lambda p: (p.team == self.team and p.type == 2, 1)
        own_team_three = lambda p: (p.team == self.team and p.type == 3, 1)
        own_team_ten = lambda p: (p.team == self.team and p.type == 10, 1)
        # own_team = lambda p: (p.team == self.team and p.can_move, p.type)
        own_team_flag = lambda p: (p.team == self.team and not p.can_move, 1)

        opp_team_one = lambda p: (p.team == self.other_team and p.type == 1, 1)
        opp_team_two = lambda p: (p.team == self.other_team and p.type == 2, 1)
        opp_team_three = lambda p: (p.team == self.other_team and p.type == 3, 1)
        opp_team_ten = lambda p: (p.team == self.other_team and p.type == 10, 1)
        # opp_team = lambda p: (p.team == self.other_team and p.can_move, p.type)
        opp_team_flag = lambda p: (p.team == self.other_team and not p.can_move, 1)
        obstacle = lambda p: (p.type == 99, 1)
        # return own_team, own_team_flag, opp_team, opp_team_flag # flag is also bomb
        return own_team_one, own_team_two, own_team_three, own_team_ten, own_team_flag, \
               opp_team_one, opp_team_two, opp_team_three, opp_team_ten, opp_team_flag, obstacle


class FourPiecesBomb(Reinforce):
    def __init__(self, team):
        super(FourPiecesBomb, self).__init__(team=team)
        self.action_dim = 28  # 16 (for piece 2) + 3 * 4 (for pieces 1, 3, 10)
        self.state_dim = len(self.state_represent())
        self.model = models.FourPiecesBomb(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/fourpieces.pkl'))

    def state_represent(self):
        own_team_one = lambda p: (p.team == self.team and p.type == 1, 1)
        own_team_two = lambda p: (p.team == self.team and p.type == 2, 1)
        own_team_three = lambda p: (p.team == self.team and p.type == 3, 1)
        own_team_ten = lambda p: (p.team == self.team and p.type == 10, 1)
        own_team_bombs = lambda p: (p.team == self.team and p.type == 11, 1)
        own_team_flag = lambda p: (p.team == self.team and p.type == 0, 1)

        # opp_team = lambda p: (p.team == self.other_team and p.can_move, p.type)
        opp_team_one = lambda p: (p.team == self.other_team and p.type == 1, 1)
        opp_team_two = lambda p: (p.team == self.other_team and p.type == 2, 1)
        opp_team_three = lambda p: (p.team == self.other_team and p.type == 3, 1)
        opp_team_ten = lambda p: (p.team == self.other_team and p.type == 10, 1)
        opp_team_bombs = lambda p: (p.team == self.other_team and p.type == 11, 1)
        opp_team_flag = lambda p: (p.team == self.other_team and p.type == 0, 1)
        obstacle = lambda p: (p.type == 99, 1)
        # return own_team, own_team_flag, opp_team, opp_team_flag # flag is also bomb
        return own_team_one, own_team_two, own_team_three, own_team_ten, own_team_flag, own_team_bombs, \
               opp_team_one, opp_team_two, opp_team_three, opp_team_ten, opp_team_flag, opp_team_bombs, obstacle
        # return own_team_one, own_team_two, own_team_three, own_team_ten, own_team_flag, own_team_bombs, \
        #        opp_team, opp_team_flag, opp_team_bombs, obstacle

class Stratego(Reinforce):
    def __init__(self, team):
        super(Stratego, self).__init__(team=team)
        self.action_dim = 64  # all pieces 3 * 16 (for pieces: 2, 2, 2) + 4 * 4 for (for pieces 1, 3, 3, 10)
        self.state_dim = len(self.state_represent())
        self.model = models.ThreePieces(self.state_dim, self.action_dim)
        # self.model.load_state_dict(torch.load('./saved_models/stratego.pkl'))

    def state_represent(self):
        own_team_one = lambda p: (p.team == self.team and p.type == 1, 1)
        own_team_two_1 = lambda p: (p.team == self.team and p.type == 2 and p.version == 1, 1)
        own_team_two_2 = lambda p: (p.team == self.team and p.type == 2 and p.version == 2, 1)
        own_team_two_3 = lambda p: (p.team == self.team and p.type == 2 and p.version == 3, 1)
        own_team_three_1 = lambda p: (p.team == self.team and p.type == 3 and p.version == 1, 1)
        own_team_three_2 = lambda p: (p.team == self.team and p.type == 3 and p.version == 2, 1)
        own_team_ten = lambda p: (p.team == self.team and p.type == 10, 1)
        own_team_flag = lambda p: (p.team == self.team and not p.type == 0, 1)
        own_team_bomb_1 = lambda p: (p.team == self.team and not p.type == 11 and p.version == 1, 1)
        own_team_bomb_2 = lambda p: (p.team == self.team and not p.type == 11 and p.version == 2, 1)

        opp_team_one = lambda p: (p.team == self.other_team and p.type == 1, 1)
        opp_team_two_1 = lambda p: (p.team == self.other_team and p.type == 2 and p.version == 1, 1)
        opp_team_two_2 = lambda p: (p.team == self.other_team and p.type == 2 and p.version == 2, 1)
        opp_team_two_3 = lambda p: (p.team == self.other_team and p.type == 2 and p.version == 3, 1)
        opp_team_three_1 = lambda p: (p.team == self.other_team and p.type == 3 and p.version == 1, 1)
        opp_team_three_2 = lambda p: (p.team == self.other_team and p.type == 3 and p.version == 2, 1)
        opp_team_ten = lambda p: (p.team == self.other_team and p.type == 10, 1)
        opp_team_flag = lambda p: (p.team == self.other_team and not p.type == 0, 1)
        opp_team_bomb_1 = lambda p: (p.team == self.other_team and not p.type == 11 and p.version == 1, 1)
        opp_team_bomb_2 = lambda p: (p.team == self.other_team and not p.type == 11 and p.version == 2, 1)
        obstacle = lambda p: (p.type == 99, 1)
        #return own_team, own_team_flag, opp_team, opp_team_flag, obstacle # flag is also bomb
        return own_team_one, own_team_two_1, own_team_two_2, own_team_two_3, own_team_three_1, own_team_three_2,\
                own_team_ten, own_team_flag, own_team_bomb_1, own_team_bomb_2,\
                opp_team_one, opp_team_two_1, opp_team_two_2, opp_team_two_3, opp_team_three_1, opp_team_three_2,\
                opp_team_ten, opp_team_flag, opp_team_bomb_1, opp_team_bomb_2, obstacle


class MiniMax(Agent):
    """
    Agent deciding his moves based on the minimax algorithm. The agent guessed the enemies setup
    before making a decision by using the current information available about the pieces.
    """
    def __init__(self, team, setup=None, depth=None):
        super(MiniMax, self).__init__(team=team, setup=setup)
        # rewards for planning the move
        self.kill_reward = 10  # killing an enemy piece
        self.neutral_fight = 2  # a neutral outcome of a fight
        self.winGameReward = 100  # finding the enemy flag
        self.certainty_multiplier = 1.2  # killing known, not guessed, enemy pieces

        # initial maximum depth of the minimax algorithm
        self.ext_depth = depth
        self.max_depth = 2  # standard max depth

        # the matrix table for deciding battle outcomes between two pieces
        self.battleMatrix = helpers.get_battle_matrix()

    def decide_move(self):
        """
        Depending on the amount of enemy pieces left, we are entering the start, mid or endgame
        and planning through the minimax algorithm.
        :return: tuple of tuple positions representing the move
        """
        if self.ext_depth is None:
            self.set_max_depth()  # set max_depth each turn
        else:
            self.max_depth = self.ext_depth
        # make sure a flag win will be discounted by a factor that guarantees a preference towards immediate flag kill
        self.winGameReward = max(self.winGameReward, self.max_depth*self.kill_reward)
        return self.minimax(max_depth=self.max_depth)

    def set_max_depth(self):
        n_alive_enemies = sum([True for piece in self.ordered_opp_pieces if not piece.dead])
        # TODO is this really doing what it should?
        if n_alive_enemies <= 1:
            self.max_depth = 2
        elif 3 <= n_alive_enemies <= 5:
            self.max_depth = 4
        elif 5 < n_alive_enemies <= 10:
            self.max_depth = 6

    def minimax(self, max_depth):
        """
        given the maximum depth, copy the known board so far, assign the pieces by random, while still
        respecting the current knowledge, and then decide the move via minimax algorithm.
        :param max_depth: int
        :return: tuple of position tuples
        """
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        # this is what the expectimax agent will think

        # get my possible actions, then shuffle them to ensure randomness when no action
        # stands out as the best
        my_doable_actions = helpers.get_poss_moves(board, self.team)
        np.random.shuffle(my_doable_actions)

        # check for terminal-state scenario
        done, won = self.goal_test(my_doable_actions, board, max_val=True)
        if done or depth == 0:
            return current_reward + self.get_terminal_reward(done, won, depth), None

        val = -float('inf')
        best_action = None
        for action in my_doable_actions:
            board, fight_result = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward + self.add_temp_reward(fight_result)
            new_val = self.min_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val < new_val:
                val = new_val
                best_action = action
            if val >= beta:
                board = self.undo_last_move(board)  # TODO why is this needed?
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
            board = self.undo_last_move(board)
        return val, best_action

    def min_val(self, board, current_reward, alpha, beta, depth):
        # this is what the opponent will think, the min-player

        # get my possible actions, then shuffle them to ensure randomness when no action
        # stands out as the best
        my_doable_actions = helpers.get_poss_moves(board, self.other_team)
        np.random.shuffle(my_doable_actions)

        # check for terminal-state scenario or maximum depth
        done, won = self.goal_test(my_doable_actions, board, max_val=False)
        if done or depth == 0:
            return current_reward + self.get_terminal_reward(done, won, depth), None

        val = float('inf')  # initial value set, so min comparison later possible
        best_action = None
        # iterate through all actions
        for action in my_doable_actions:
            board, fight_result = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            temp_reward = current_reward + self.add_temp_reward(fight_result)
            new_val = self.max_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val > new_val:
                val = new_val
                best_action = action
            if val <= alpha:
                board = self.undo_last_move(board)  # TODO why is this needed?
                return val, best_action
            beta = min(beta, val)
            board = self.undo_last_move(board)
        return val, best_action

    def add_temp_reward(self, fight_result):
        # depending on the fight we want to update the current paths value
        temp_reward = 0
        if fight_result is not None:
            if fight_result == 1:
                temp_reward = self.kill_reward
            elif fight_result == 2:
                temp_reward = int(self.certainty_multiplier * self.kill_reward)
            elif fight_result == 0:
                temp_reward = self.neutral_fight  # both pieces die
            elif fight_result == -1:
                temp_reward = -self.kill_reward
            elif fight_result == -2:
                temp_reward = -int(self.certainty_multiplier * self.kill_reward)
        return temp_reward

    def goal_test(self, actions_possible, board, max_val):
        """
        check the board for whether a flag has been captured already and return the winning game rewards,
        if not check whether there are no actions possible anymore, return TRUE then, or FALSE.
        :param actions_possible: list of moves
        :param board: numpy array (5, 5)
        :return: boolean: reached terminal state, boolean: own team (True) or other team won (False)
        """
        # if board is not None:
        flag_alive = [False, False]
        for pos, piece in np.ndenumerate(board):
            if piece is not None and piece.type == 0:
                flag_alive[piece.team] = True
        if not flag_alive[self.other_team]:
            return True, True
        if not flag_alive[self.team]:
            return True, False
        # else:             # TODO deprecate this, board should not be None
        #     if 0 in self.deadPieces[0] or 0 in self.deadPieces[1]:
        #         # print('flag captured')
        #         return True
        if not actions_possible:  # TODO no reward for winning with killing all enemies?
            # print('cannot move anymore')
            if max_val:  # TODO think about, if this order is correct
                won = False
            else:
                won = True  # empirically False 65 : 35, 63 : 37 vs  True: 78 : 22, 65 : 35, 71 : 29 against Random
            return True, won
        else:
            return False, None

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return 0
        else:
            if won:
                terminal_reward = self.winGameReward
            elif not won:
                terminal_reward = - self.winGameReward
            return terminal_reward * (depth + 1) / (self.max_depth + 1) * (terminal_reward / self.kill_reward)

    def update_prob_by_fight(self, enemy_piece):
        """
        update the information about the given piece, after a fight occured
        :param enemy_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        enemy_piece.potential_types = [enemy_piece.type]

    def update_prob_by_move(self, move, moving_piece):
        """
        update the information about the given piece, after it did the given move
        :param move: tuple of positions tuples
        :param moving_piece: object of class Piece
        :return: change is in-place, no value specified
        """
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            moving_piece.hidden = False
            moving_piece.potential_types = [moving_piece.type]  # piece is 2
        else:
            immobile_enemy_types = [idx for idx, type in enumerate(moving_piece.potential_types)
                                    if type in [0, 11]]
            moving_piece.potential_types = np.delete(moving_piece.potential_types, immobile_enemy_types)

    def draw_consistent_enemy_setup(self, board):
        """
        Draw a setup of the enemies pieces on the board provided that aligns with the current status of
        information about said pieces, then place them on the board. This is done via iterative random sampling,
        until a consistent draw occurs. This draw may or may not represent the overall true distribution of the pieces.
        :param board: numpy array (5, 5)
        :return: board with the assigned enemy pieces in it.
        """
        # get information about enemy pieces (how many, which alive, which types, and indices in assign. array)
        enemy_pieces = copy.deepcopy(self.ordered_opp_pieces)
        enemy_pieces_alive = [piece for piece in enemy_pieces if not piece.dead]
        types_alive = [piece.type for piece in enemy_pieces_alive]

        # do the following as long as the drawn assignment is not consistent with the current knowledge about them
        consistent = False
        sample = None
        while not consistent:
            # choose as many pieces randomly as there are enemy pieces alive
            sample = np.random.choice(types_alive, len(types_alive), replace=False)
            # while-loop break condition
            consistent = True
            for idx, piece in enumerate(enemy_pieces_alive):
                # if the drawn type doesn't fit the potential types of the current piece, then redraw
                if sample[idx] not in piece.potential_types:
                    consistent = False
                    break
        # place this draw now on the board by assigning the types and changing critical attributes
        for idx, piece in enumerate(enemy_pieces_alive):
            # add attribute of the piece being guessed (only happens in non-real gameplay aka planning)
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
        """
        Undo the last move in the memory. Return the updated board.
        :param board: numpy array (5, 5)
        :return: board
        """
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


class Omniscient(MiniMax):
    """
    Child of ExpectiSmart agent. This agent is omniscient and thus knows the location and type of each
    piece of the enemy. It then plans by doing a minimax algorithm.
    """
    def __init__(self, team, setup=None, depth=None):
        super(Omniscient, self).__init__(team=team, setup=setup, depth=depth)
        self.setup = setup
        self.winFightReward = 10
        self.neutralFightReward = 5
        self.winGameReward = 1000

    def install_board(self, board):
        super().install_board(board)
        self.unhide_all()

    def unhide_all(self):
        """
        Uncover all enemy pieces by setting the hidden attribute to False
        :return: None
        """
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


class Heuristic(MiniMax):
    def __init__(self, team, setup=None, depth=2):
        super(Heuristic, self).__init__(team=team, setup=setup, depth=depth)
        self.evaluator = ThreePieces(team)
        self.winGameReward = 100

    def install_board(self, board):
        super().install_board(board)
        self.evaluator.install_board(board)
        # self.unhide_all()

    def get_network_reward(self):
        state = self.evaluator.board_to_state()
        state_action_values = self.evaluator.model(Variable(state, volatile=True)).data.numpy()
        return np.max(state_action_values)

    def get_terminal_reward(self, done, won, depth):
        if not done:
            return self.get_network_reward()
        else:
            if won:
                terminal_reward = self.winGameReward
            elif not won:
                terminal_reward = - self.winGameReward
            return terminal_reward * (depth + 1) / (self.max_depth + 1) * (terminal_reward / self.kill_reward)

