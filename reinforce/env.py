from matplotlib import pyplot as plt
import numpy as np
import copy as cp
import random
import torch  # TODO: eliminate torch dependencies?


import pieces
import game


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
        self.own_pieces = []
        self.opp_pieces = []
        pieces_list = self.decide_pieces()
        for p in pieces_list:
            if p.team == 0:
                self.own_pieces.append(p)
            else:
                self.opp_pieces.append(p)
        choices = np.random.choice(len(positions), len(pieces_list), replace=False)
        chosen_pos = []
        for i in choices:
            chosen_pos.append(positions[i])
        for p in pieces_list:
            self.board[chosen_pos.pop()] = p

        self.board[(2, 2)] = pieces.Piece(99, 99)  # place obstacle

        self.score = 0
        self.reward = 0
        self.steps = 0
        self.goal_test = False

    def reset(self):  # resetting means freshly initializing
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

    def get_team_pos(self, team):
        team_pos = []
        for pos in self.board_positions:
            piece = self.board[pos]
            if piece is not None:
                if piece.team == team:  # own
                    team_pos.append(pos)
        return team_pos

    def step(self, action):
        self.reward = 0
        self.agent_step(action)
        self.opponent_step()

        self.score += self.reward
        self.steps += 1
        return self.reward, self.goal_test

    def agent_step(self, action):
        raise NotImplementedError

    def opponent_step(self):
        raise NotImplementedError

    def show(self):
        fig = plt.figure(1)
        game.print_board(self.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


class FindFlag(Env):
    def __init__(self):
        super(FindFlag, self).__init__()

    def decide_pieces(self):
        pieces_list = [pieces.Piece(3, 0), pieces.Piece(10, 1), pieces.Piece(10, 1), pieces.Piece(0, 1)]
        return pieces_list

    def agent_step(self, action):
        i_piece = int(action / 4)  # which piece: 0-3 is first 4-7 second etc.
        agent = self.own_pieces[i_piece]
        agent_pos = self.find_piece(agent)  # where is the piece
        action = action % 4
        go_to_pos = self.action_to_pos(action, agent_pos)

        if go_to_pos not in self.board_positions:
            self.reward += -1  # hitting the wall
        else:
            piece = self.board[go_to_pos]
            if piece is not None:
                if piece.type == 99:
                    self.reward += -1  # hitting obstacle
                if piece.type == 0:
                    self.reward += 10
                    self.goal_test = True
            else:
                self.board[go_to_pos] = agent  # move to position
                self.board[agent_pos] = None
                # reward += -0.01 * self.steps  # each step more and more difficult
                self.reward += -0.1

    def opponent_step(self):
        pass
        # self.piece_move(self.enemy_pos1)
        # self.piece_move(self.enemy_pos2)

    def piece_move(self, piece_pos):
        opp_piece = self.board[piece_pos]
        opp_action = random.randint(0, 3)
        go_to_pos = self.action_to_pos(opp_action, piece_pos)
        if go_to_pos in self.board_positions:
            piece = self.board[go_to_pos]
            if piece is not None:
                # if piece.type == 99:
                #     pass  # hitting obstacle
                # if piece.type == 0:
                #     pass  # cannot capture own flag
                if piece.type == 3:
                    self.reward -= 1  # kill agent
                    self.goal_test = True  # agent died
            else:
                self.board[go_to_pos] = opp_piece  # move to position
                self.board[piece_pos] = None

    def action_to_pos(self, action, init_pos):
        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        direction = moves[action]  # action: 0, 1, 2, 3
        go_to_pos = [sum(x) for x in zip(init_pos, direction)]  # go in this direction
        go_to_pos = tuple(go_to_pos)
        return go_to_pos

    def find_piece(self, piece):
        for pos in self.board_positions:
            if self.board[pos] == piece:
                return pos
            else:
                print("Error: Piece not found!")
