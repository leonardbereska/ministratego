from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import random
import torch
import pieces
import game
import battleMatrix
# import agents
import six


class Env:
    """
    Environment: 5x5 board, agent piece, opponents flag piece.
    Rewards: Big negative reward for impossible action (hitting wall/obstacle)
            and small negative for every action not finding the flag.
    State: Complete board
    """

    def __init__(self, board_size=(5, 5)):

        self.board = np.empty(board_size, dtype=object)
        self.board_positions = [(i, j) for i in range(board_size[0]) for j in range(board_size[1])]
        positions = cp.deepcopy(self.board_positions)

        # place obstacles
        obstacle_pos = self.decide_obstacles()
        for o in obstacle_pos:
            positions.remove(o)  # remove obstacle positions from possible piece positions
            self.board[o] = pieces.Piece(99, 99)  # place obstacles

        self.living_pieces = [[], []]  # team 0,  team 1
        self.dead_pieces = [[], []]
        known_pieces, random_pieces = self.decide_pieces()

        # place known pieces
        for (p, pos) in known_pieces:
            self.board[pos] = p
            positions.remove(pos)
            self.living_pieces[p.team].append(p)
        # place random pieces
        c = list(np.random.choice(len(positions), len(random_pieces), replace=False))
        for p in random_pieces:
            self.board[positions[c.pop()]] = p
            self.living_pieces[p.team].append(p)

        self.fight = battleMatrix.get_battle_matrix()

        self.opp_can_move = False  # static opponent would be e.g. only flag
        for p in self.living_pieces[1]:  # if movable piece among opponents pieces
            if p.can_move:
                self.opp_can_move = True

        self.score = 0
        self.reward = 0
        self.steps = 0

        # rewards (to be overriden by subclass)
        self.reward_illegal = 0  # punish illegal moves
        self.reward_step = 0  # negative reward per agent step
        self.reward_win = 0  # win game
        self.reward_loss = 0  # lose game
        self.reward_kill = 0  # kill enemy figure reward
        self.reward_die = 0  # lose to enemy figure

    def reset(self):  # resetting means freshly initializing
        self.__init__()

    def decide_pieces(self):
        raise NotImplementedError

    # def decide_board(self):
    #     pass  # TODO: non-random initial board setting

    # def decide_agents(self):
    #     raise NotImplementedError

    def decide_obstacles(self):  # standard: obstacle in middle
        obstacle_pos = [(2, 2)]
        return obstacle_pos

    def get_state(self):
        raise NotImplementedError

    def step(self, action):
        self.reward = 0
        agent_move = self.action_to_move(action, team=0)
        if not game.is_legal_move(self.board, agent_move):
            self.reward += self.reward_illegal
            self.score += self.reward
            return self.reward, False  # environment does not change, agent should better choose only legal moves
        self.do_move(agent_move, team=0)

        if self.opp_can_move:
            opp_move = self.random_move(team=1)
            self.do_move(opp_move, team=1)  # assuming only legal moves selected

        self.steps += 1
        done = self.goal_test()
        self.score += self.reward
        return self.reward, done

    def random_move(self, team):      # can be overwritten to give smarter opponent
        moves = game.get_poss_actions(self.board, team)
        if not moves:
            return None  # no move possible
        move = random.choice(moves)
        return move

    def do_move(self, move, team):
        if move is None:
            return
        other_team = (team + 1) % 2
        pos_from, pos_to = move

        piece_to = self.board[pos_to]
        piece_from = self.board[pos_from]

        if piece_to is not None:
            outcome = self.fight[piece_from.type, piece_to.type]
            if outcome == -1:  # lose
                self.dead_pieces[team].append(piece_from)
                self.board[pos_from] = None
                if team == 0:
                    self.reward += self.reward_die
            elif outcome == 0:  # tie
                self.dead_pieces[team].append(piece_from)
                self.dead_pieces[other_team].append(piece_to)
                self.board[pos_from] = None
                self.board[pos_to] = None
            elif outcome == 1:  # win
                self.dead_pieces[other_team].append(piece_to)
                self.board[pos_from] = None
                self.board[pos_to] = piece_from
                if team == 0:
                    self.reward += self.reward_kill
        else:
            self.board[pos_to] = piece_from  # move to position
            self.board[pos_from] = None
            if team == 0:
                self.reward += self.reward_step

    def action_to_move(self, action, team):
        i = int(action / 4)  # which piece: 0-3 is first 4-7 second etc.
        piece = self.living_pieces[team][i]
        piece_pos = self.find_piece(piece)  # where is the piece
        if piece_pos is None:
            move = (None, None)  # return illegal move
            return move
        action = action % 4  # 0-3 as direction

        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # a piece can move in four directions
        direction = moves[action]  # action: 0-3
        pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
        pos_to = tuple(pos_to)
        move = (piece_pos, pos_to)
        return move

    def find_piece(self, piece):
        for pos in self.board_positions:
            if self.board[pos] == piece:
                return pos
        print("Error: Piece not found!")

    def goal_test(self):
        for p in self.dead_pieces[1]:
            if p.type == 0:
                self.reward += self.reward_win
                return True
        for p in self.dead_pieces[0]:
            if p.type == 0:
                self.reward += self.reward_loss
                return True
        if self.opp_can_move:  # only if opponent is playing, killing his pieces wins
            if not game.get_poss_actions(self.board, team=1):
                self.reward += self.reward_win
                return True
        if not game.get_poss_actions(self.board, team=0):
            self.reward += self.reward_loss
            return True
        if self.score < -10:
            self.reward += self.reward_loss
            print("lost")
            return True
        return False

    def show(self):
        fig = plt.figure(1)
        game.print_board(self.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


class FindFlag(Env):
    def __init__(self):
        super(FindFlag, self).__init__()
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10

    def decide_pieces(self):
        known_place = [(pieces.Piece(0, 1), (4, 4))]
        random_place = [pieces.Piece(3, 0)]
        return known_place, random_place

    def get_state(self):
        state_dim = 3
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:  # opponents team
                    board_state[tuple([1] + list(pos))] = 1  # flag
                else:
                    board_state[tuple([2] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Escape(Env):
    def __init__(self):
        super(Escape, self).__init__()
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10
        self.reward_loss = -1

    def decide_pieces(self):
        known_place = []
        random_place = [pieces.Piece(3, 0), pieces.Piece(3, 0),
                       pieces.Piece(11, 1), pieces.Piece(10, 1),
                       pieces.Piece(3, 1), pieces.Piece(10, 1),
                       pieces.Piece(0, 1)]
        return known_place, random_place

    def get_state(self):
        state_dim = 4
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:  # opponents team
                    if p.type == 0:
                        board_state[tuple([1] + list(pos))] = 1  # flag
                    else:
                        board_state[tuple([2] + list(pos))] = 1  # opp piece
                else:
                    board_state[tuple([3] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Maze(Env):
    def __init__(self):
        super(Maze, self).__init__()
        # self.reward_step
        self.reward_illegal = -1
        self.reward_win = 10

    def decide_pieces(self):
        known_place = [(pieces.Piece(3, 0), (0, 0)), (pieces.Piece(0, 1), (4, 4))]
        random_place = []
        return known_place, random_place

    def decide_obstacles(self):
        return [(3, 1), (3, 2), (3, 3), (3, 4), (1, 0), (1, 1), (1, 2), (1, 3)]

    def get_state(self):
        state_dim = 3
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:
                    board_state[tuple([1] + list(pos))] = 1  # flag
                else:
                    board_state[tuple([2] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


# class Kill(Env):
#     def __init__(self):
#         super(Kill, self).__init__()
#         self.reward_step = -0.1
#         self.reward_illegal = -1
#         self.reward_win = 10
#         self.reward_kill = 1
#
#     def decide_pieces(self):
#         pieces_list = [pieces.Piece(10, 0), pieces.Piece(3, 1), pieces.Piece(3, 1)]
#         return pieces_list
#
#
# class SmallGame(Env):
#     def __init__(self):
#         super(SmallGame, self).__init__()
#         self.reward_step = -0.1
#         self.reward_illegal = -1
#         self.reward_win = 10
#
#     def decide_pieces(self):
#         pieces_list = [pieces.Piece(3, 0), pieces.Piece(0, 0), pieces.Piece(3, 1), pieces.Piece(0, 1)]
#         return pieces_list
#
#
# class MiniStratego(Env):
#     def __init__(self):
#         super(MiniStratego, self).__init__()
#         self.reward_step = -0.1
#         self.reward_illegal = -1
#         self.reward_win = 10
#
#     def decide_pieces(self):
#         pass

    # TODO: fixed placement of pieces
    # TODO: agent integration

# TODO: state representation
# max 7 channels
# own: movable,
#      immovable
# opp: known, movable
#             bomb
#      unknown, moved
#               not_moved
# obstacles

# own
# opp

# obstacles yes/no?
# own movable/immovable?
# opp movable/immovable?
# opp known/unknown?
