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
        if type in (0, 11, 88, 99):
            self.can_move = False
            self.move_radius = 0
        elif type == 2:
            self.can_move = True
            self.move_radius = float('Inf')
        else:
            self.can_move = True
            self.move_radius = 1

    def __repr__(self):  # for printing pieces on the board return type of piece
        if self.type == 0:
            return "f"
        if self.type == 11:
            return "b"
        if self.type == 88:
            return "?"
        if self.type == 99:
            return "X"
        else:
            return str(self.type)

    def change_hidden(self, new_status):
        self.hidden = new_status
