"""
Pieces
"""
import numpy as np


class Piece:
    def __init__(self, type, team, position):
        self.position = position
        #self.positions_history = [position]
        self.unique_identifier = np.random.randint(0, 10000)
        self.dead = False
        self.hidden = True
        assert(type in (0, 1, 2, 3, 10, 11, 88, 99))  # 0: flag, 11: bomb, 88: unknown, 99: obstacle
        self.type = type
        assert(team == 0 or team == 1 or team == 99)  # 99 is a neutral piece: e.g. obstacle
        self.team = team
        self.has_moved = False
        if type in (0, 11, 88, 99):
            self.can_move = False
            self.move_radius = 0
        elif type == 2:
            self.can_move = True
            self.move_radius = float('Inf')
        else:
            self.can_move = True
            self.move_radius = 1

        # each entry of this dict is a list containting the probability P_k of hidden piece j being piece k, i.e.
        # oppPiecesProbabilites[3,0] = [P_0, P_1, P_2, P_3, P_10, P_11] with indices declaring k
        # this is important as long as the piece is _hidden_
        self.piece_probabilites = dict()
        self.piece_probabilites[0] = 0.1
        self.piece_probabilites[1] = 0.1
        self.piece_probabilites[2] = 0.3
        self.piece_probabilites[3] = 0.2
        self.piece_probabilites[10] = 0.1
        self.piece_probabilites[11] = 0.2

    # CAVEAT: Turning this specific form into __repr__ fucks up numpy arrays that hold objects of class pieces
    #         thus we either need to read further into __repr__ in combo with numpy or leave it as __str__ atm
    def __str__(self):  # for printing pieces on the board return type of piece
        if self.hidden:
            return "?"
        else:
            if self.type == 0:
                return "f"
            if self.type == 11:
                return "b"
            if self.type == 99:
                return "X"
            else:
                return str(self.type)

    def change_position(self, new_pos):
        #self.positions_history.append(new_pos)
        self.position = new_pos
        self.has_moved = True

