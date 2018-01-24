import copy

import numpy as np

import battleMatrix
import helpers
import pieces


class Game:
    def __init__(self, agent0, agent1):
        """
        player1 = Player()
        player2 = Player()
        self.board = np.zeros((5, 5))
        self.board[2, 2] = None  # obstacle in the middle
        self.board[0:1, 0:5] = player1.decideInitialSetup()
        self.board[3:4, 0:5] = player2.decideInitialSetup()
        """

        self.agents = (agent0, agent1)
        self.board = np.empty((5, 5), dtype=object)

        self.types_available = np.array([0, 1, 2, 2, 2, 3, 3, 10, 11, 11])
        setup0, setup1 = agent0.setup, agent1.setup
        agent0.install_opp_setup(copy.deepcopy(setup1))
        agent1.install_opp_setup(copy.deepcopy(setup0))

        for idx, piece in np.ndenumerate(setup0):
            piece.hidden = False
            self.board[piece.position] = piece
        for idx, piece in np.ndenumerate(setup1):
            piece.hidden = False
            self.board[piece.position] = piece
        obstacle = pieces.Piece(99, 99, (2, 2))
        obstacle.hidden = False
        self.board[2, 2] = obstacle  # set obstacle

        self.move_count = 1  # agent 1 starts

        self.deadPieces = []
        dead_piecesdict = dict()
        for type_ in set(self.types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def run_game(self):
        game_over = False
        rewards = None
        while not game_over:
            helpers.print_board(self.board)
            rewards = self.run_step()
            if rewards is not None:
                game_over = True
        return rewards

    def run_step(self):
        turn = self.move_count % 2  # player 1 or player 0
        print("Round: " + str(self.move_count))
        for agent_ in self.agents:
            agent_.move_count = self.move_count

        if self.move_count > 1000:  # if game lasts longer than 1000 turns => tie
            return 0, 0  # each agent gets reward 0
        new_move = self.agents[turn].decide_move()
        # test if agent can't move anymore
        if new_move is None:
            if turn == 1:
                return 2, -2  # agent0 wins
            elif turn == 0:
                return -2, 2  # agent1 wins
        self.do_move(new_move)  # execute agent's choice
        # test if game is over
        if self.goal_test():  # flag discovered
            if turn == 1:
                return -1, 1  # agent1 wins
            elif turn == 0:
                return 1, -1  # agent0 wins
        self.move_count += 1
        return None

    def do_move(self, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        # let agents update their boards too
        for _agent in self.agents:
            _agent.do_move(move, true_gameplay=True)

        if not helpers.is_legal_move(self.board, move):
            return False  # illegal move chosen
        self.board[from_].has_moved = True
        if not self.board[to_] is None:  # Target field is not empty, then has to fight
            fight_outcome = self.fight(self.board[from_], self.board[to_])
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, self.board[from_]))
                self.update_board((from_, None))
            elif fight_outcome == 0:
                self.update_board((to_, None))
                self.update_board((from_, None))
            else:
                self.update_board((from_, None))
                self.update_board((to_, self.board[to_]))
        else:
            self.update_board((to_, self.board[from_]))
            self.update_board((from_, None))

        return True

    def update_board(self, updated_piece):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        if piece is not None:
            piece.change_position(pos)
        self.board[pos] = piece
        return

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadPieces[piece_def.team][piece_def.type] += 1
        elif outcome == 0:
            self.deadPieces[piece_def.team][piece_def.type] += 1
            self.deadPieces[piece_att.team][piece_att.type] += 1
        elif outcome == -1:
            self.deadPieces[piece_att.team][piece_att.type] += 1
        return outcome

    def is_legal_move(self, move_to_check):  # TODO: redirect all references to this function to helpers
        return helpers.is_legal_move(self.board, move_to_check)

    def goal_test(self):
        if self.deadPieces[0][0] == 1 or self.deadPieces[1][0] == 1:
            # print('flag captured')
            return True
        else:
            return False

