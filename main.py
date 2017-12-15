import numpy as np
import random
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
        self.deadFigures.append(([]))
        self.deadFigures.append(([]))

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
        self.battleMatrix[10, 1] = -1
        self.battleMatrix[10, 2] = 1
        self.battleMatrix[10, 3] = 1
        self.battleMatrix[10, 10] = 2

    def run_game(self):
        while True:
            visible_state, actions_possible = self.get_visible_state()
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
            if self.goal_test():
                break
            self.turn += 1
        if self.turn % 2 == 1:
            return 1  # player1 won!
        else:
            return 0  # player2 won

    def do_move(self, move):
        if not self.is_legal_move(move):
            return 0  # illegal move chosen
        if not self.board[move[1]] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(self.board[move[0]], self.board[move[1]])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return -1
            elif fight_outcome == 1:
                self.board[move[1]] = self.board[move[0]]
                self.board[move[0]] = None
            elif fight_outcome == 2:
                self.board[move[1]] = None
                self.board[move[0]] = None
            else:
                self.board[move[0]] = None

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
        """
        Checks if move is legal
        """
        pos_before = moveToCheck[0]
        pos_after = moveToCheck[1]

        if self.board[pos_before] is None:
            return False  # no piece on field to move
        elif self.board[pos_before].type in [0, 11]:
            return False  # bomb and flag cant move
        if abs(pos_after[0] - pos_before[0]) + abs(pos_after[1] - pos_before[1]) > 1:  # move across more than one field
            if not pos_before[0] == pos_after[0] and not pos_before[1] == pos_after[1]:
                return False  # no diagonal moves allowed
            if not self.board[pos_before].type == 2:
                return False  # piece on field cant move more than 1 field (not a 2)
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
        elif not actions_possible
        else:
            return False

    def get_visible_state(self):
        """
        :return: a deepcopy of the board with pieces visible to the agent, and his possible actions
        """
        board_visible = copy.deepcopy(self.board)
        same_team = self.turn % 2
        other_team = (self.turn + 1) % 2
        actions_possible = []

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
                                # TODO: use piece.move_radius to restrict search
                                move = (pos, pos_to)
                                if self.is_legal_move(move):
                                    actions_possible.append(move)
            print(pos)

        return board_visible, actions_possible


new = Game()
new.run_game()
