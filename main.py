import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt
import pieces
import agent
import copy
import pickle


class Game:
    def __init__(self, setup):
        """
        player1 = Player()
        player2 = Player()
        self.board = np.zeros((5, 5))
        self.board[2, 2] = None  # obstacle in the middle
        self.board[0:1, 0:5] = player1.decideInitialSetup()
        self.board[3:4, 0:5] = player2.decideInitialSetup()
        """
        self.agents = (agent.SmartSetup(0, setup), agent.RandomAgent(1))
        self.board = np.empty((5, 5), dtype=object)

        self.types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        setup0 = self.agents[0].decide_setup(self.types_available)
        setup1 = self.agents[1].decide_setup(self.types_available)
        setup1 = np.flip(setup1, 0)  # flip setup for second player
        self.board[3:5, 0:5] = setup0
        self.board[0:2, 0:5] = setup1
        self.board[2, 2] = pieces.Piece(99, 99)  # set obstacle

        self.move_count = 1  # agent 1 starts

        self.deadFigures = []
        self.deadFigures.append([])
        self.deadFigures.append([])

        self.battleMatrix = dict()
        self.battleMatrix[1, 11] = -1
        self.battleMatrix[1, 1] = 0
        self.battleMatrix[1, 2] = -1
        self.battleMatrix[1, 3] = -1
        self.battleMatrix[1, 0] = 1
        self.battleMatrix[1, 10] = 1
        self.battleMatrix[2, 0] = 1
        self.battleMatrix[2, 11] = -1
        self.battleMatrix[2, 1] = 1
        self.battleMatrix[2, 2] = 0
        self.battleMatrix[2, 3] = -1
        self.battleMatrix[2, 10] = -1
        self.battleMatrix[3, 0] = 1
        self.battleMatrix[3, 11] = 1
        self.battleMatrix[3, 2] = 1
        self.battleMatrix[3, 3] = 0
        self.battleMatrix[3, 1] = 1
        self.battleMatrix[3, 10] = -1
        self.battleMatrix[10, 0] = 1
        self.battleMatrix[10, 11] = -1
        self.battleMatrix[10, 1] = 1
        self.battleMatrix[10, 2] = 1
        self.battleMatrix[10, 3] = 1
        self.battleMatrix[10, 10] = 0

    def run_game(self):
        game_over = False
        rewards = None
        while not game_over:
            rewards = self.run_step()
            if rewards is not None:
                game_over = True
        return rewards

    def run_step(self):
        visible_state, actions_possible = self.get_agent_knowledge()
        turn = self.move_count % 2  # player 1 or player 0

        # test if game is over
        if self.goal_test(actions_possible):  # flag already discovered or no action possible
            if turn == 1:
                return 1, -1  # agent0 wins
            elif turn == 0:
                return -1, 1  # agent1 wins
        if self.move_count > 1000:  # if game lasts longer than 1000 turns => tie
            return 0, 0  # each agent gets reward 0

        new_move = self.agents[turn].decide_move(visible_state, actions_possible)
        self.do_move_proxy(new_move)  # execute agent's choice
        self.move_count += 1
        return None

    def do_move_proxy(self, move):
        """
        replaces do_move function with working proxy until update_moves is implemented
        assumes move has been checked to be legal previously
        """
        from_pos = move[0]
        to_pos = move[1]
        from_piece = self.board[from_pos]
        to_piece = self.board[to_pos]

        if to_piece is None:  # if field empty then simply move there
            self.board[to_pos] = from_piece
            self.board[from_pos] = None
        else:  # if not empty field => fight
            outcome = self.fight(from_piece, to_piece)
            if outcome == 1:  # attacker wins
                self.board[from_pos] = None
                self.board[to_pos] = from_piece
            elif outcome == -1:  # defendant wins
                self.board[from_pos] = None
                self.board[to_pos] = to_piece  # piece stays
            elif outcome == 0:  # tie: both die
                self.board[from_pos] = None
                self.board[to_pos] = None

    def do_move(self, move):
        """
        :param move: tuple or array consisting of coordinates from in 0 and to in 1
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
                self.update_board((to_, self.board[from_]), True)
                self.update_board((from_, None), True)
            elif fight_outcome == 0:
                self.update_board((to_, None), True)
                self.update_board((from_, None), True)
            else:
                self.update_board((from_, None), True)
        else:
            self.update_board([(from_, None), (to_, self.board[from_])], False)
        return True

    def update_board(self, updated_pieces, visible):
        """
        :param updated_pieces: array of tuples (piece_board_position, piece_object)
        :param visible: boolean, True if the piece is visible to the enemy team, False if hidden
        :return: void
        """
        if visible:
            self.agent0.updateBoard(updated_pieces)
            self.agent1.updateBoard(updated_pieces)
        else:
            if not updated_pieces[1] is None:
                if updated_pieces[1].team == 0:
                    self.agent0.updateBoard(updated_pieces)
                    self.agent1.updateBoard((updated_pieces[0], pieces.Piece(88, 1)))
                else:
                    self.agent0.updateBoard((updated_pieces[0], pieces.Piece(88, 0)))
                    self.agent1.updateBoard(updated_pieces)
            else:
                self.agent0.updateBoard(updated_pieces)
                self.agent1.updateBoard(updated_pieces)

        for pos, piece in updated_pieces:
            self.board[pos] = piece

    def fight(self, piece_att, piece_def):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if outcome == 1:
            self.deadFigures[piece_def.team].append(piece_def.type)
        elif outcome == 0:
            self.deadFigures[piece_def.team].append(piece_def.type)
            self.deadFigures[piece_att.team].append(piece_att.type)
        elif outcome == -1:
            self.deadFigures[piece_att.team].append(piece_att.type)
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
        if not self.board[pos_after] is None:
            if self.board[pos_after].team == self.board[pos_before].team:
                return False  # cant fight own pieces
            if self.board[pos_after].type == 99:
                return False  # cant fight obstacles
        return True

    def goal_test(self, actions_possible):
        if 0 in self.deadFigures[0] or 0 in self.deadFigures[1]:
            # print('flag captured')
            return True
        elif not actions_possible:
            # print('cannot move anymore')
            return True
        else:
            return False

    def get_agent_knowledge(self):
        """
        :return: a deepcopy of the board with pieces visible to the agent, and his possible actions
        """
        board_visible = copy.deepcopy(self.board)
        same_team = self.move_count % 2
        other_team = (self.move_count + 1) % 2
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
    """
    Plots a board object in a pyplot figure
    """
    board = copy.deepcopy(board)  # ensure to not accidentally change input
    plt.interactive(False)  # make plot stay? true: close plot, false: keep plot
    fig = plt.figure()
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
            plt.gca().invert_yaxis()  # own pieces down, others up
            # transpose pos[0], pos[1] to turn board
            plt.plot(pos[1], pos[0], piece_marker, markersize=37)  # plot markers for pieces
            plt.annotate(str(piece), xy=(pos[1], pos[0]), size=20, ha="center", va="center")  # piece type on marker
    plt.show()
    return fig


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
            new = Game(setup)
            # if simu % 10 == 0:
            #     print('\nTotal rewards: {}, Simulation {}/{}'.format(total_reward, simu, num_simu))
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


