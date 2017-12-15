import numpy as np
import scipy
import random
from matplotlib import pyplot as plt

import pieces
import agent
import copy


class Game:
    def __init__(self):
        """
        player1 = Player()
        player2 = Player()
        self.board = np.zeros((5, 5))
        self.board[2, 2] = None  # obstacle in the middle
        self.board[0:1, 0:5] = player1.decideInitialSetup()
        self.board[3:4, 0:5] = player2.decideInitialSetup()
        """
        self.agent0 = agent.Agent(0)
        self.agent1 = agent.Agent(1)

        self.board = np.empty((5, 5), dtype=object)

        self.types_available = [1, 2, 2, 2, 3, 3, 10, 11, 11, 0]
        setup0 = self.agent0.decide_setup(self.types_available)
        setup1 = self.agent1.decide_setup(self.types_available)
        setup1 = np.flip(setup1, 0)  # flip setup for second player
        self.board[3:5, 0:5] = setup0
        self.board[0:2, 0:5] = setup1
        self.board[2, 2] = pieces.Piece(99, 99)  # initialize obstacle

        self.turn = 1

        self.deadFigures = []
        self.deadFigures.append([])
        self.deadFigures.append([])

        self.battleMatrix = dict()
        self.battleMatrix[1, 11] = -1
        self.battleMatrix[1, 1] = 2
        self.battleMatrix[1, 2] = -1
        self.battleMatrix[1, 3] = -1
        self.battleMatrix[1, 10] = 1
        self.battleMatrix[2, 0] = 1
        self.battleMatrix[2, 11] = -1
        self.battleMatrix[2, 1] = 1
        self.battleMatrix[2, 2] = 2
        self.battleMatrix[2, 3] = -1
        self.battleMatrix[2, 10] = -1
        self.battleMatrix[3, 0] = 1
        self.battleMatrix[3, 11] = 1
        self.battleMatrix[3, 2] = 1
        self.battleMatrix[3, 3] = 2
        self.battleMatrix[3, 1] = 11
        self.battleMatrix[3, 10] = -1
        self.battleMatrix[10, 0] = 1
        self.battleMatrix[10, 11] = -1
        self.battleMatrix[10, 1] = 1
        self.battleMatrix[10, 2] = 1
        self.battleMatrix[10, 3] = 1
        self.battleMatrix[10, 10] = 2

    def run_game(self):
        Tie = False
        while True:
            visible_state, actions_possible = self.get_agent_knowledge()
            if not actions_possible:
                return (self.turn + 1) % 2  # other player won
            if self.turn % 2 == 1:
                new_move = self.agent0.decide_move(visible_state, actions_possible)
            else:
                new_move = self.agent1.decide_move(visible_state, actions_possible)
            if not self.is_legal_move(new_move):
                print('Warning, player chose illegal move!')
            else:
                self.do_move(new_move)
            print(self.board)
            decision = self.goal_test()
            if decision:
                break
            elif decision == 0:
                Tie = True
            if self.goal_test(actions_possible):
                break
            self.turn += 1
        if Tie:
            return 0.5
        elif self.turn % 2 == 1:
            return 1  # player1 won!
        else:
            return 0  # player2 won

    def do_move(self, move):
        '''
        :param move: tuple or array consisting of coordinates from in 0 and to in 1
        :return:
        '''
        from_ = move[0]
        to_ = move[1]
        if not self.is_legal_move(move):
            return False  # illegal move chosen
        if not self.board[to_] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(self.board[from_], self.board[to_])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.updateBoard( (to_, self.board[from_]), True)
                self.updateBoard( (from_, None), True)
            elif fight_outcome == 2:
                self.updateBoard( (to_, None), True)
                self.updateBoard( (from_, None), True)
            else:
                self.updateBoard( (from_, None), True)
        else:
            self.updateBoard( [(from_, None), (to_, self.board[from_])], False)
        return True

    def updateBoard(self, updatedPieces, visible):
        '''
        :param updatedPieces: array of tuples (piece_board_position, piece_object)
        :param visible: boolean, True if the piece is visible to the enemy team, False if hidden
        :return: void
        '''
        if visible:
            self.agent0.updateBoard(updatedPieces)
            self.agent1.updateBoard(updatedPieces)
        else:
            if not updatedPieces[1] is None:
                if updatedPieces[1].team == 0:
                    self.agent0.updateBoard(updatedPieces)
                    self.agent1.updateBoard( (updatedPieces[0], pieces.Piece(88, 1)) )
                else:
                    self.agent0.updateBoard( (updatedPieces[0], pieces.Piece(88, 0)) )
                    self.agent1.updateBoard(updatedPieces)
            else:
                self.agent0.updateBoard(updatedPieces)
                self.agent1.updateBoard(updatedPieces)

        for pos, piece in updatedPieces:
            self.board[pos] = piece

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces
        """
        if piece_att.team == piece_def.team:
            return None  # same team cant fight
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadFigures[piece_def.team].append(piece_def.type)
        elif outcome == 2:
            self.deadFigures[piece_def.team].append(piece_def.type)
            self.deadFigures[piece_att.team].append(piece_att.type)
        else:
            self.deadFigures[piece_att.team].append(piece_att.type)
        return outcome

    def is_legal_move(self, moveToCheck):
        '''

        :param moveToCheck: array/tuple with the coordinates of the position from and to
        :return: True if warrants a legal move, False if not
        '''
        pos_before = moveToCheck[0]
        pos_after = moveToCheck[1]

        if self.board[pos_before] is None:
            return False  # no piece on field to move
        move_dist = scipy.spatial.distance.cityblock(pos_before, pos_after)
        if move_dist > self.board[pos_before].move_radius:
            return False  # move too far for selected piece
        if move_dist > 1:
            if not pos_before[0] == pos_after[0] and not pos_before[1] == pos_after[1]:
                return False  # no diagonal moves allowed
            else:
                if pos_after[0] == pos_before[0]:
                    dist_sign = int(np.sign(pos_after[1] - pos_before[1]))
                    for k in list(range(pos_before[1] + dist_sign, pos_after[1], int(dist_sign))):
                        if self.board[(pos_before[0], k)] is not None:
                            return False  # pieces in the way of the move
                else:
                    dist_sign = int(np.sign(pos_after[0] - pos_before[0]))
                    for k in range(pos_before[0] + dist_sign, pos_after[0], int(dist_sign)):
                        if self.board[(k, pos_before[1])] is not None:
                            return False  # pieces in the way of the move
        if not self.board[pos_after] is None:
            if self.board[pos_after].team == self.board[pos_before].team:
                return False  # cant fight own pieces
        return True

    def goal_test(self, actions_possible):
        if 0 in self.deadFigures[0] or 0 in self.deadFigures[1]:
            return True
        elif not actions_possible:
            return True
        #TODO: implement tie checkup to avoid near endless loops

        else:
            return False

    def get_agent_knowledge(self):
        """
        :return: a deepcopy of the board with pieces visible to the agent, and his possible actions
        """
        board_visible = copy.deepcopy(self.board)
        same_team = self.turn % 2
        other_team = (self.turn + 1) % 2
        actions_possible = []
        # TODO: CLean up visible states update and only leave possible actions calc behind
        for pos in ((i, j) for i in range(5) for j in range(5)):
            piece = board_visible[pos]  # select a piece for all possible board positions
            if piece is not None:  # board positions has a piece on it
                if piece.type != 99:  # that piece is not an obstacle
                    if piece.team == other_team:
                        # check if piece is hidden or not
                        if piece.hidden:
                            board_visible[pos] = pieces.Piece(88, other_team)  # replace piece at board with unknown
                    elif piece.team == same_team:
                        # check which moves are possible
                        if piece.can_move:
                            for pos_to in ((i, j) for i in range(5) for j in range(5)):
                                move = (pos, pos_to)
                                if self.is_legal_move(move):
                                    actions_possible.append(move)
        return board_visible, actions_possible


def print_board(board):
    board = copy.deepcopy(board)
    plt.figure()
    Z1 = np.add.outer(range(5), range(5)) % 2  # board
    plt.imshow(Z1, cmap=plt.cm.magma, alpha=.5, interpolation='nearest')
    for pos in ((i, j) for i in range(5) for j in range(5)):
        # place pieces on board
        piece = board[pos]
        if piece is not None:
            if piece.team == 1:
                color = 'b'
            elif piece.team == 0:
                color = 'r'
            else:  # must be obstacle
                color = 'k'
            if piece.can_move:
                form = 'o'
            else:  # either not move or unknown
                form = 's'
            if piece.type == 0:  # if flag
                form = 'X'
            piecemarker = ''.join(('-', color, form))
            plt.gca().invert_yaxis()
            plt.plot(pos[1], pos[0], piecemarker, markersize=37)  # transpose pos[0], pos[1]
            plt.annotate(str(piece), xy=(pos[1], pos[0]), size=20, ha="center", va="center")
    plt.show()


new = Game()
print_board(new.board)
# new.run_game()
