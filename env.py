import copy as cp

import numpy as np
import torch
from matplotlib import pyplot as plt


import helpers
import pieces


class Env:
    """
    Environment superclass
    """

    def __init__(self, agent0, agent1, board_size=(5, 5)):

        # initialize board
        self.board = np.empty(board_size, dtype=object)
        self.board_positions = [(i, j) for i in range(board_size[0]) for j in range(board_size[1])]
        positions = cp.deepcopy(self.board_positions)

        # place obstacles
        obstacle_pos = self.decide_obstacles()
        for o_pos in obstacle_pos:
            positions.remove(o_pos)  # remove obstacle positions from possible piece positions
            self.board[o_pos] = pieces.Piece(99, 99, o_pos)  # place obstacles

        self.living_pieces = [[], []]  # team 0,  team 1
        self.dead_pieces = [[], []]

        known_pieces, random_pieces = self.decide_pieces()
        typecounter = [dict(), dict()]
        for piece in known_pieces + random_pieces:
            typecounter[piece.team][piece.type] = 0
        # place fixed position pieces
        for piece in known_pieces:
            self.board[piece.position] = piece
            positions.remove(piece.position)
            self.living_pieces[piece.team].append(piece)
            typecounter[piece.team][piece.type] += 1
            piece.version = typecounter[piece.team][piece.type]
        # place random position pieces
        c = list(np.random.choice(len(positions), len(random_pieces), replace=False))
        for p in random_pieces:
            i_pos = c.pop()
            p.position = positions[i_pos]
            self.board[p.position] = p
            self.living_pieces[p.team].append(p)
            typecounter[piece.team][piece.type] += 1
            piece.version = typecounter[piece.team][piece.type]

        # give agents board
        agent0.install_board(self.board)
        agent1.install_board(self.board)
        self.agents = (agent0, agent1)

        # give agent actors (pieces whose movements he controls)
        for team in (0, 1):
            actors = []
            for pos in self.board_positions:
                p = self.board[pos]
                if p is not None:
                    if p.team == team:
                        if p.can_move:
                            actors.append(p)
            actors = sorted(actors, key=lambda actor: actor.type + actor.version/10)
            # train for unique actor sequence, sort by type and version
            self.agents[team].action_represent(actors)

        self.battleMatrix = helpers.get_battle_matrix()

        self.opp_can_move = False  # static opponent would be e.g. only flag
        for p in self.living_pieces[1]:  # if movable piece among opponents pieces
            if p.can_move:
                self.opp_can_move = True

        self.score = 0
        self.reward = 0
        self.steps = 0
        self.death_thresh = None
        self.illegal_moves = 0

        # rewards (to be overridden by subclass environment)
        self.reward_illegal = 0  # punish illegal moves
        self.reward_step = 0  # negative reward per agent step
        self.reward_win = 0  # win game
        self.reward_loss = 0  # lose game
        self.reward_kill = 0  # kill enemy figure reward
        self.reward_die = 0  # lose to enemy figure
        # self.reward_iter = 0  # no iteration TODO deprecate this

    def reset(self):  # resetting means freshly initializing
        self.__init__(agent0=self.agents[0], agent1=self.agents[1])

    def decide_pieces(self):
        raise NotImplementedError

    def decide_obstacles(self):  # standard: obstacle in middle
        obstacle_pos = [(2, 2)]
        return obstacle_pos

    def step(self, move=None):
        """
        Perform one step of the environment: agents in turn choose a move
        :param move: externally determined move to be performed by agent0 (useful for training)
        :return: reward accumulated in this step, boolean: if environment in terminal state, boolean: if agent0 won
        """
        self.reward = 0
        self.steps += 1  # illegal move as step

        # are there still pieces to be moved?
        if not helpers.get_poss_moves(self.board, team=0):
            self.reward += self.reward_loss
            self.score += self.reward
            return self.reward, True, False  # 0 for lost

        # move decided by agent or externally?
        if move is not None:
            agent_move = move  # this enables working with the environment in external functions (e.g. train.py)
        else:
            agent_move = self.agents[0].decide_move()

        # is move legal?
        if not helpers.is_legal_move(self.board, agent_move):  # if illegal -> no change in env, receive reward_illegal
            self.reward += self.reward_illegal
            self.illegal_moves += 1
            # print("Warning: agent 1 selected an illegal move: {}".format(agent_move))
            self.score += self.reward
            done, won = self.goal_test()
            return self.reward, done, won  # environment does not change for illegal

        self.do_move(agent_move, team=0)

        # opponents move
        if self.opp_can_move:  # only if opponent is playing, killing his pieces wins (opponent can be e.g. flag only)

            # are there still pieces to be moved?
            if not helpers.get_poss_moves(self.board, team=1):
                self.reward += self.reward_win
                self.score = self.reward
                return self.reward, True, True  # 1 for won
            opp_move = self.agents[1].decide_move()

            # is move legal?
            if not helpers.is_legal_move(self.board, opp_move):  # opponent is assumed to only perform legal moves
                pass
                # print("Warning: agent 1 selected an illegal move: {}".format(opp_move))

            self.do_move(opp_move, team=1)  # assuming only legal moves selected

        done, won = self.goal_test()
        self.score += self.reward
        return self.reward, done, won

    # def action_to_move(self, action, team):  # TODO deprecate this (is in agent now)
    #     i = int(np.floor(action / 4))  # which piece: 0-3 is first 4-7 second etc.
    #     piece = self.living_pieces[team][i]
    #     piece_pos = piece.position  # where is the piece
    #     if piece_pos is None:
    #         move = (None, None)  # return illegal move
    #         return move
    #     action = action % 4  # 0-3 as direction
    #     moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # a piece can move in four directions
    #     direction = moves[action]  # action: 0-3
    #     pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
    #     pos_to = tuple(pos_to)
    #     move = (piece_pos, pos_to)
    #     return move

    def do_move(self, move, team):
        if move is None:  # no move chosen (network)?
            return
        if not helpers.is_legal_move(self.board, move):
            return False  # illegal move chosen
        other_team = (team + 1) % 2
        pos_from, pos_to = move
        piece_from = self.board[pos_from]
        piece_to = self.board[pos_to]

        # agents updating their board too
        for _agent in self.agents:
            _agent.do_move(move, true_gameplay=True)

        piece_from.has_moved = True
        if piece_to is not None:  # Target field is not empty, then has to fight
            fight_outcome = self.battleMatrix[piece_from.type, piece_to.type]

            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((pos_to, piece_from))
                self.update_board((pos_from, None))
                self.dead_pieces[other_team].append(piece_to)
                if team == 0:
                    self.reward += self.reward_kill
            elif fight_outcome == 0:
                self.update_board((pos_to, None))
                self.update_board((pos_from, None))
                self.dead_pieces[team].append(piece_from)
                self.dead_pieces[other_team].append(piece_to)
            elif fight_outcome == -1:
                self.update_board((pos_from, None))
                self.update_board((pos_to, piece_to))
                self.dead_pieces[team].append(piece_from)
                if team == 0:
                    self.reward += self.reward_die

        else:
            self.update_board((pos_to, piece_from))
            self.update_board((pos_from, None))
            if team == 0:
                self.reward += self.reward_step

        return True

    def update_board(self, updated_piece):
        pos = updated_piece[0]
        piece = updated_piece[1]
        if piece is not None:
            piece.change_position(pos)  # adapt position for piece
        self.board[pos] = piece  # place piece on board position
        return

    def goal_test(self):
        """
        Check if the game is in a terminal state due to flag capture
        (note: in env.step it is already checked if there are still pieces to move)
        :return: (bool: is environment in a terminal state, bool: is it won (True) or lost (False)
        """
        for p in self.dead_pieces[1]:
            if p.type == 0:
                self.reward += self.reward_win
                # print("Red won, captured flag")
                return True, True
        for p in self.dead_pieces[0]:
            if p.type == 0:
                self.reward += self.reward_loss
                # print("Blue won, captured flag")
                return True, False
        if self.death_thresh is not None:
            if self.score < self.death_thresh:
                self.reward += self.reward_loss
                return True, False
        return False, False

    def show(self):
        fig = plt.figure(1)
        helpers.print_board(self.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


################################################################################################################
# Subclasses

class FindFlag(Env):
    def __init__(self, agent0, agent1):
        super(FindFlag, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_win = 1

    def decide_pieces(self):
        known_place = []
        random_place = [pieces.Piece(3, 0, None), pieces.Piece(0, 1, None)]
        return known_place, random_place


class Maze(Env):
    def __init__(self, agent0, agent1):
        super(Maze, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_illegal = -1
        self.reward_win = 1
        # self.reward_iter = -1
        # self.reward_loss = -1

    def decide_pieces(self):
        known_place = [pieces.Piece(0, 1, (4, 4))]
        random_place = [pieces.Piece(3, 0, None)]
        return known_place, random_place

    def decide_obstacles(self):
        return [(3, 1), (3, 2), (3, 3), (3, 4), (1, 0), (1, 1), (1, 2), (1, 3)]



class TwoPieces(Env):
    def __init__(self, agent0, agent1):
        super(TwoPieces, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step = -0.1
        # self.reward_illegal = -0.5
        self.reward_kill = 0.2
        # self.reward_die = 0
        # self.reward_loss = 0
        self.reward_win = 1
        # self.death_thresh = -20

    def decide_pieces(self):
        self.types_available = [0, 1, 10]
        known_place = []
        # draw random setup for 10 figures for each team
        for team in (0, 1):
            setup_pos = [(i, j) for i in range(team * 3, 2 + team * 3) for j in range(5)]
            index = np.random.choice(range(len(setup_pos)), len(setup_pos), replace=False)
            for i, piece_type in enumerate(self.types_available):
                # print(setup_pos[index[i]])
                known_place.append(pieces.Piece(piece_type, team, setup_pos[index[i]]))
        random_place = []  # random_place is across whole board
        return known_place, random_place


class ThreePieces(Env):
    def __init__(self, agent0, agent1):
        super(ThreePieces, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step = -0.1  # only important in self-play to escape stale-mates
        # self.reward_illegal = -0.1  # no illegal moves allowed
        # self.reward_kill = 0
        # self.reward_die = -1
        # self.reward_loss = -1
        self.reward_win = 1
        # self.death_thresh = -20

    def decide_pieces(self):
        self.types_available = [[0, 1, 3, 10], [0, 1, 3, 10]]
        known_place = []
        # draw random setup for 10 figures for each team
        for team in (0, 1):
            setup_pos = [(i, j) for i in range(team * 3, 2 + team * 3) for j in range(5)]
            index = np.random.choice(range(len(setup_pos)), len(setup_pos), replace=False)
            for i, piece_type in enumerate(self.types_available[team]):
                # print(setup_pos[index[i]])
                known_place.append(pieces.Piece(piece_type, team, setup_pos[index[i]]))
        random_place = []  # random_place is across whole board
        return known_place, random_place


class FourPieces(Env):
    def __init__(self, agent0, agent1):
        super(FourPieces, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step = -0.1  # only important in self-play to escape stale-mates
        # self.reward_illegal = -0.1  # no illegal moves allowed
        self.reward_kill = 0
        self.reward_die = -0
        self.reward_loss = 0
        self.reward_win = 1
        # self.death_thresh = -20

    def decide_pieces(self):
        self.types_available = [[0, 1, 2, 3, 10], [0, 1, 2, 3, 10]]
        known_place = []
        # draw random setup for 10 figures for each team
        for team in (0, 1):
            setup_pos = [(i, j) for i in range(team * 3, 2 + team * 3) for j in range(5)]
            index = np.random.choice(range(len(setup_pos)), len(setup_pos), replace=False)
            for i, piece_type in enumerate(self.types_available[team]):
                # print(setup_pos[index[i]])
                known_place.append(pieces.Piece(piece_type, team, setup_pos[index[i]]))
        random_place = []  # random_place is across whole board
        return known_place, random_place


class Stratego(Env):
    def __init__(self, agent0, agent1):
        super(Stratego, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step = -0.1
        # self.reward_kill = 0
        # self.reward_die = -0
        # self.reward_illegal = 0
        self.reward_win = 1
        # self.reward_loss = 0

    def decide_pieces(self):
        self.types_available = np.array([0, 1, 2, 2, 2, 3, 3, 10, 11, 11])  # has to be here for correct order
        known_place = []
        # draw random setup for 10 figures for each team
        for team in (0, 1):
            setup_pos = [(i, j) for i in range(team * 3, 2 + team * 5) for j in range(5)]
            setup = np.random.choice(self.types_available, len(self.types_available), replace=False)
            for i, pos in enumerate(setup):
                known_place.append(pieces.Piece(setup[i], team, setup_pos[i]))
        random_place = []  # random_place is across whole board
        return known_place, random_place


# def watch_game(env, step_time):  # TODO deprecate this -> main
#     """
#     Watch two agents play against each other, step_time is
#     """
#     new_game = env
#     done = False
#     while not done:
#         move = 0
#         _, done = new_game.step(move)
#         # print(env.reward)
#         new_game.show()
#         plt.pause(step_time)
#
#     if env.reward > 0:  # reward_win is for Red
#         outcome = "Red won!"
#     else:
#         outcome = "Blue won!"
#     print(outcome)
#     plt.title(outcome)
#     plt.show(block=True)  # keep plot



# TODO how does agent deal with uncertainty? how does he master controlling pieces of same value?
# TODO -> adapt state representation to this list: