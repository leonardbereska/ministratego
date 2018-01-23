import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import copy
import torch
from collections import namedtuple
import random


def is_legal_move(board, move_to_check):
    """
    :param move_to_check: array/tuple with the coordinates of the position from and to
    :return: True if move is a legal move, False if not
    """
    pos_before = move_to_check[0]
    pos_after = move_to_check[1]
    if pos_after not in [(i, j) for i in range(5) for j in range(5)]:
        return False
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


def print_board(board):
    """
    Plots a board object in a pyplot figure
    """
    board = copy.deepcopy(board)  # ensure to not accidentally change input
    plt.interactive(False)  # make plot stay? true: close plot, false: keep plot
    plt.figure(1)
    plt.clf()
    layout = np.add.outer(range(5), range(5)) % 2  # chess-pattern board
    plt.imshow(layout, cmap=plt.cm.magma, alpha=.5, interpolation='nearest')  # plot board
    for pos in ((i, j) for i in range(5) for j in range(5)):  # go through all board positions
        piece = board[pos]  # select piece on respective board position
        # decide which marker type to use for piece
        if piece is not None:
            piece.hidden = False  # not the best but ensures everything is printed without "?"

            if piece.team == 1:
                color = 'b'  # blue: player 1
            elif piece.team == 0:
                color = 'r'  # red: player 0
            else:
                color = 'k'  # black: obstacle
            if piece.can_move:
                form = 'o'  # circle: for movable
            else:
                form = 's'  # square: either immovable or unknown piece
            if piece.type == 0:
                form = 'X'  # cross: flag
            piece_marker = ''.join(('-', color, form))
            plt.plot(pos[1], pos[0], piece_marker, markersize=37)  # plot markers for pieces
            plt.annotate(str(piece), xy=(pos[1], pos[0]), size=20, ha="center", va="center")  # piece type on marker
    #plt.gca().invert_yaxis()  # own pieces down, others up
    plt.pause(0.01)
    plt.show(block=False)


def get_poss_moves(board, team):
    """
    :return: List of possible actions for agent of team
    """
    actions_possible = []
    for pos, piece in np.ndenumerate(board):
        if piece is not None:  # board position has a piece on it
            if not piece.type == 99:  # that piece is not an obstacle
                if piece.team == team:
                    # check which moves are possible
                    if piece.can_move:
                        for pos_to in ((i, j) for i in range(5) for j in range(5)):
                            move = (pos, pos_to)
                            if is_legal_move(board, move):
                                actions_possible.append(move)
    return actions_possible


def plot_scores(episode_scores, n_smooth):
    plt.figure(2)
    plt.clf()
    scores_t = torch.FloatTensor(episode_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    average = [0]
    if len(scores_t) >= n_smooth:
        means = scores_t.unfold(0, n_smooth, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(n_smooth-1), means))
        average = means.numpy()
        plt.plot(average)
    plt.title('Average Score over last {} Episodes: {}'.format(n_smooth, int(average[-1]*10)/10))
    plt.pause(0.001)  # pause a bit so that plots are updated


class ReplayMemory(object):
    """
    Stores a state-transition (s, a, s', r) quadruple
    for approximating Q-values with q(s, a) <- r + gamma * max_a' q(s', a') updates
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))