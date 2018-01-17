from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import random
import torch  # TODO: eliminate torch dependencies?


import pieces
import game
import battleMatrix


class Env:
    """
    Environment: 5x5 board, agent piece, opponents flag piece.
    Rewards: Big negative reward for impossible action (hitting wall/obstacle)
            and small negative for every action not finding the flag.
    State: Complete board
    """

    def __init__(self):
        self.board = np.empty((5, 5), dtype=object)
        self.board_positions = [(i, j) for i in range(5) for j in range(5)]
        positions = cp.deepcopy(self.board_positions)
        positions.remove((2, 2))  # remove obstacle position

        # randomly select positions for participating pieces
        self.living_pieces = [[], []]
        self.dead_pieces = [[], []]
        pieces_list = self.decide_pieces()
        for p in pieces_list:
            if p.team == 0:
                self.living_pieces[0].append(p)
            else:
                self.living_pieces[1].append(p)
        choices = np.random.choice(len(positions), len(pieces_list), replace=False)
        chosen_pos = []
        for i in choices:
            chosen_pos.append(positions[i])
        for p in pieces_list:
            self.board[chosen_pos.pop()] = p

        self.board[(2, 2)] = pieces.Piece(99, 99)  # place obstacle
        self.fight = battleMatrix.get_battle_matrix()

        self.opp_can_move = False
        for p in self.living_pieces[1]:  # if movable piece among opponents pieces
            if p.can_move:
                self.opp_can_move = True

        self.score = 0
        self.reward = 0
        self.steps = 0

        self.reward_illegal = 0  # to be overwritten by subclass
        self.reward_step = 0
        self.reward_win = 0
        self.reward_loss = 0

    def reset(self):  # resetting means freshly initializing TODO: subclass or superclass?
        self.__init__()

    def decide_pieces(self):
        raise NotImplementedError

    def get_state(self):
        """
        Get state representation for decision network
        return: full board: 5x5xstate_dim Tensor one-hot Tensor of board with own, opponents figures
        """
        state_dim = 4
        board_state = np.zeros((state_dim, 5, 5))  # 5x5 board with channels: 0: own, 1: obstacles, 2: opponent
        for pos in self.board_positions:
            if self.board[pos] is not None:  # piece on this field
                if self.board[pos].team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = 1
                elif self.board[pos].team == 1:  # opponents team
                    if self.board[pos].type == 0:  # flag
                        board_state[tuple([2] + list(pos))] = 1
                    elif self.board[pos].team == 1:  # enemy
                        board_state[tuple([3] + list(pos))] = 1
                else:  # obstacle piece
                    board_state[tuple([1] + list(pos))] = 1
        board_tensor = torch.FloatTensor(board_state)
        board_tensor = board_tensor.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_tensor

    def step(self, action):
        self.reward = 0
        agent_move = self.action_to_move(action, team=0)
        if not game.is_legal_move(self.board, agent_move):
            self.reward += self.reward_illegal
            self.score += self.reward
            return self.reward, False  # environment does not change, agent should better choose only legal moves
        self.do_move(agent_move, team=0)

        if self.opp_can_move:
            opp_move = self.opponent_move()
            if game.is_legal_move(self.board, opp_move):
                self.do_move(opp_move, team=1)

        self.steps += 1
        done = self.goal_test()
        self.score += self.reward
        return self.reward, done

    # can be overwritten to give smarter opponent
    def opponent_move(self):
        actions = game.get_poss_actions(self.board, 1)
        move = random.choice(actions)
        # self.action_to_move(action, team=1)
        return move

    def do_move(self, move, team):
        other_team = (team + 1) % 2
        pos_from, pos_to = move

        piece_to = self.board[pos_to]
        piece_from = self.board[pos_from]

        if piece_to is not None:
            outcome = self.fight[piece_from.type, piece_to.type]
            if outcome == -1:  # lose
                self.dead_pieces[team].append(piece_from)
                self.board[pos_from] = None
            elif outcome == 0:  # tie
                self.dead_pieces[team].append(piece_from)
                self.dead_pieces[other_team].append(piece_to)
                self.board[pos_from] = None
                self.board[pos_to] = None
            elif outcome == 1:  # win
                self.dead_pieces[other_team].append(piece_to)
                self.board[pos_from] = None
                self.board[pos_to] = piece_from
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

    def find_piece(self, piece):  # TODO: test! this will most likely not work
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
        self.reward_win = 1
        self.reward_loss = -1

    def decide_pieces(self):
        pieces_list = [pieces.Piece(3, 0), pieces.Piece(0, 1)]
        return pieces_list


class Escape(Env):
    def __init__(self):
        super(Escape, self).__init__()
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10
        self.reward_loss = -1

    def decide_pieces(self):
        pieces_list = [pieces.Piece(3, 0), pieces.Piece(3, 0), pieces.Piece(11, 1), pieces.Piece(10, 1), pieces.Piece(3, 1), pieces.Piece(10, 1), pieces.Piece(0, 1)]
        return pieces_list



