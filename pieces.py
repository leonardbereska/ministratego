"""
Pieces
"""


class Piece:
    def __init__(self, type, team):
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

    # CAVEAT: Turning this specific form into __repr__ fucks up numpy arrays that hold objects of class pieces
    #         thus we either need to further read into __repr__ in combo with numpy or leave it as __str__ for the moment
    # TODO: UNDO CHANGE OF STRING OUTPUT FOR DEBUG
    def __str__(self):  # for printing pieces on the board return type of piece
        if self.type == 0:
            return str(self.team) + "/f"
        if self.type == 11:
            return str(self.team) + "/b"
        if self.type == 88:
            return str(self.team) + "/?"
        if self.type == 99:
            return str(self.team) + "/X"
        else:
            return str(self.team) + "/" + str(self.type)

    def set_status_hidden(self, new_status):
        self.hidden = new_status
        return self

    def set_status_has_moved(self, new_status):
        self.has_moved = new_status
        return self


class unknownPiece(Piece):
    def __init__(self, team):
        super().__init__(88, team)
        self.has_moved = False
        # each entry of this dict is a list containting the probability P_k of hidden piece j being piece k, i.e.
        # oppPiecesProbabilites[3,0] = [P_0, P_1, P_2, P_3, P_10, P_11] with indices declaring k
        self.piece_probabilites = dict()
        self.piece_probabilites[0] = 0.1
        self.piece_probabilites[1] = 0.1
        self.piece_probabilites[2] = 0.3
        self.piece_probabilites[3] = 0.2
        self.piece_probabilites[10] = 0.1
        self.piece_probabilites[11] = 0.2
