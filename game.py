import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import pieces
import agent
import copy
import pickle
import battleMatrix


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
        setup0 = self.agents[0].decide_setup(self.types_available)
        setup1 = self.agents[1].decide_setup(self.types_available)
        # setup1 = np.flip(setup1, 0)  # flip setup for second player
        # this positioning of agent 0 and agent 1 on the board is now hardcoded!! Dont change
        self.board[3:5, 0:5] = setup0
        self.board[0:2, 0:5] = setup1
        self.board[2, 2] = pieces.Piece(99, 99)  # set obstacle

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
            print_board(self.board)
            rewards = self.run_step()
            if rewards is not None:
                game_over = True
        return rewards

    def run_step(self):
        turn = self.move_count % 2  # player 1 or player 0
        # print("Round: " + str(self.move_count))

        # test if game is over
        if self.goal_test():  # flag already discovered or no action possible
            if turn == 1:
                return 1, -1  # agent0 wins
            elif turn == 0:
                return -1, 1  # agent1 wins
        if self.move_count > 1000:  # if game lasts longer than 1000 turns => tie
            return 0, 0  # each agent gets reward 0
        new_move = self.agents[turn].decide_move()
        if new_move == (None, None):  # agent cant move anymore --> lost
            if turn == 1:
                return 1, -1  # agent0 wins
            elif turn == 0:
                return -1, 1  # agent1 wins
        self.do_move(new_move)  # execute agent's choice
        self.move_count += 1
        return None

    def do_move(self, move):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
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
                self.update_board((to_, self.board[from_]), visible=True)
                self.update_board((from_, None), visible=True)
            elif fight_outcome == 0:
                self.update_board((to_, None), visible=True)
                self.update_board((from_, None), visible=True)
            else:
                self.update_board((from_, None), visible=True)
                self.update_board((to_, self.board[to_]), visible=True)
        else:
            self.update_board((to_, self.board[from_]), visible=False)
            self.update_board((from_, None), visible=False)

        return True

    def update_board(self, updated_piece, visible):
        """
        :param updated_piece: tuple (piece_board_position, piece_object)
        :param visible: boolean, True if the piece is visible to the enemy team, False if hidden
        :return: void
        """
        pos = updated_piece[0]
        piece = updated_piece[1]
        if visible:
            self.agents[0].updateBoard(updated_piece)
            self.agents[1].updateBoard(updated_piece)
        else:
            if piece is not None:
                if self.agents[0].omniscient:
                    if self.agents[1].omniscient:
                        # 0 and 1 is omniscient --> both know the piece
                        self.agents[0].updateBoard(updated_piece)
                        self.agents[1].updateBoard(updated_piece)
                    else:
                        # 0 is omniscient, but 1 isnt
                        if piece.team == 0:
                            # piece belongs to team 0, so 1 doesnt know it
                            self.agents[1].updateBoard((pos, pieces.unknownPiece(piece.team)))
                            self.agents[0].updateBoard(updated_piece)
                        else:
                            # piece belongs to team 1, but both know the piece, as 0 is omniscient
                            self.agents[0].updateBoard(updated_piece)
                            self.agents[1].updateBoard(updated_piece)
                elif self.agents[1].omniscient:
                    if self.agents[0].omniscient:
                        # 0 and 1 is omniscient --> both know the piece
                        self.agents[0].updateBoard(updated_piece)
                        self.agents[1].updateBoard(updated_piece)
                    else:
                        if piece.team == 1:
                            # piece belongs to team 1, so 0 doesnt know it, because not omnsicient
                            self.agents[0].updateBoard((pos, pieces.unknownPiece(piece.team)))
                            self.agents[1].updateBoard(updated_piece)
                        else:
                            # # piece belongs to team 0, but both know the piece, as 1 is omniscient
                            self.agents[0].updateBoard(updated_piece)
                            self.agents[1].updateBoard(updated_piece)
                else:
                    # none of the agents are omniscient, update by team dependency
                    if piece.team == 0:
                        self.agents[0].updateBoard(updated_piece)
                        self.agents[1].updateBoard((pos, pieces.unknownPiece(piece.team)))
                    else:
                        self.agents[0].updateBoard((pos, pieces.unknownPiece(piece.team)))
                        self.agents[1].updateBoard(updated_piece)
            else:
                self.agents[0].updateBoard(updated_piece)
                self.agents[1].updateBoard(updated_piece)
        self.board[pos] = piece

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadPieces[piece_def.team][piece_def.type] += 1
            self.agents[0].deadPieces[piece_def.team][piece_def.type] += 1
            self.agents[1].deadPieces[piece_def.team][piece_def.type] += 1
        elif outcome == 0:
            self.deadPieces[piece_def.team][piece_def.type] += 1
            self.agents[0].deadPieces[piece_def.team][piece_def.type] += 1
            self.agents[1].deadPieces[piece_def.team][piece_def.type] += 1

            self.deadPieces[piece_att.team][piece_att.type] += 1
            self.agents[0].deadPieces[piece_att.team][piece_att.type] += 1
            self.agents[1].deadPieces[piece_att.team][piece_att.type] += 1
        elif outcome == -1:
            self.deadPieces[piece_att.team][piece_att.type] += 1
            self.agents[0].deadPieces[piece_att.team][piece_att.type] += 1
            self.agents[1].deadPieces[piece_att.team][piece_att.type] += 1
        return outcome

    def is_legal_move(self, move_to_check):
        """

        :param move_to_check: array/tuple with the coordinates of the position from and to
        :return: True if warrants a legal move, False if not
        """
        pos_before = move_to_check[0]
        pos_after = move_to_check[1]
        if self.board[pos_before] is None:
            return False  # no piece on field to move
        if not self.board[pos_after] is None:
            if self.board[pos_after].team == self.board[pos_before].team:
                return False  # cant fight own pieces
            if self.board[pos_after].type == 99:
                return False  # cant fight obstacles
        move_dist = spatial.distance.cityblock(pos_before, pos_after)
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
        return True

    def get_poss_actions(self, board, player):
        """
        :param board: Fully set playboard! This function only works after enemy pieces have been assigned before!
        :return: list of possible actions for opponent
        """
        actions_possible = []
        for pos, piece in np.ndenumerate(self.board):
            if piece is not None:  # board positions has a piece on it
                if not piece.type == 99:  # that piece is not an obstacle
                    if piece.team == player:
                        # check which moves are possible
                        if piece.can_move:
                            for pos_to in ((i, j) for i in range(5) for j in range(5)):
                                move = (pos, pos_to)
                                if self.is_legal_move(move):
                                    actions_possible.append(move)
        return actions_possible

    def goal_test(self):
        turn = self.move_count % 2
        if not self.get_poss_actions(self.board, turn):  # if list of possible actions empty
            return True
        if self.deadPieces[0][0] == 1 or self.deadPieces[1][0] == 1:
            # print('flag captured')
            return True
        else:
            return False


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
            # transpose pos[0], pos[1] to turn board
            plt.plot(pos[1], pos[0], piece_marker, markersize=37)  # plot markers for pieces
            plt.annotate(str(piece), xy=(pos[1], pos[0]), size=20, ha="center", va="center")  # piece type on marker
    plt.gca().invert_yaxis()  # own pieces down, others up
    plt.show(block=False)


def simulation():
    """
    :return: tested_setups: list of setup and winning percentage
    """
    types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
    num_simulations = 100
    num_setups = 1000
    tested_setups = []

    for i in range(num_setups):  # test 100 setups
        setup = np.random.choice(types_available, 10, replace=False)
        win_count = 0

        for simu in range(num_simulations):  # simulate games
            agent_0 = agent.RandomAgent(0)
            agent_1 = agent.SmartSetup(1, setup)
            new = Game(agent_0, agent_1)
            if simu % 10 == 0:
                print('\nTotal rewards: {}, Simulation {}/{}'.format(win_count, simu, num_simulations))
            for step in range(2000):
                game_reward = new.run_step()
                if game_reward is not None:
                    if game_reward[0] == 1:  # count wins
                        win_count += 1
                    break
        tested_setups.append((setup, win_count/num_simulations))
        print('\nAgent wins {} out of {} simulations'
              '\nSetup {} of {}'.format(win_count, num_simulations, i+1, num_setups))
    return tested_setups


# setups = simulation()
# pickle.dump(setups, open('randominit2.p', 'wb'))


good_setup = np.empty((2,5), dtype=int)
good_setup[0,0] = 3
good_setup[0,1] = 11
good_setup[0,2] = 0
good_setup[0,3] = 11
good_setup[0,4] = 1
good_setup[1,0] = 2
good_setup[1,1] = 2
good_setup[1,2] = 10
good_setup[1,3] = 2
good_setup[1,4] = 3
good_setup = np.flip(good_setup, 0)

good_setup2 = np.empty((2,5), dtype=int)
good_setup2[0,0] = 3
good_setup2[0,1] = 11
good_setup2[0,2] = 0
good_setup2[0,3] = 11
good_setup2[0,4] = 1
good_setup2[1,0] = 2
good_setup2[1,1] = 2
good_setup2[1,2] = 10
good_setup2[1,3] = 2
good_setup2[1,4] = 3
#good_setup2 = np.flip(good_setup2, 0)


# please no more testing in game.py file!
# setup_agent0 = np.empty((2,5), dtype=object)
# setup_agent1 = np.empty((2,5), dtype=object)
# for pos in ((i,j) for i in range(2) for j in range(5)):
#     setup_agent0[pos] = pieces.Piece(good_setup[pos], 0)
#     setup_agent1[pos] = pieces.Piece(good_setup2[pos], 1)
#setup0 = np.flip(setup_agent0, 0)
# agent_0 = agent.OmniscientExpectiSmart(0, setup_agent0, setup_agent1)
# agent_1 = agent.OmniscientExpectiSmart(1, setup_agent1, setup_agent0)
# game = Game(agent_0, agent_1)
# result = game.run_game()
# print(result)

