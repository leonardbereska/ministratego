import copy as cp

import numpy as np
import torch
from matplotlib import pyplot as plt

import battleMatrix
import helpers
import pieces


# import train


class Env:
    """
    Environment superclass
    """

    def __init__(self, agent0, agent1, board_size=(5, 5)):
        self.Train = True  # if true can insert action in env_step

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

        # place fixed position pieces
        for piece in known_pieces:
            self.board[piece.position] = piece
            positions.remove(piece.position)
            self.living_pieces[piece.team].append(piece)
        # place random position pieces
        c = list(np.random.choice(len(positions), len(random_pieces), replace=False))
        for p in random_pieces:
            i_pos = c.pop()
            p.position = positions[i_pos]
            self.board[p.position] = p
            self.living_pieces[p.team].append(p)

        # give agents board
        agent0.board = cp.deepcopy(self.board)
        agent1.board = cp.deepcopy(self.board)
        self.agents = (agent0, agent1)

        # give agent actors
        # actors are pieces which are under control of the agent
        for team in (0, 1):
            actors = []
            for pos in self.board_positions:
                p = self.board[pos]
                if p is not None:
                    if p.team == team:
                        if p.can_move:
                            actors.append(p)
            actors = sorted(actors, key=lambda actor: actor.type)  # train for unique actor sequence
            self.agents[team].action_represent(actors)

        self.battleMatrix = battleMatrix.get_battle_matrix()

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
        self.reward_iter = 0  # no iteration

    def reset(self):  # resetting means freshly initializing
        self.__init__(agent0=self.agents[0], agent1=self.agents[1])

    def decide_pieces(self):
        raise NotImplementedError

    def decide_obstacles(self):  # standard: obstacle in middle
        obstacle_pos = [(2, 2)]
        return obstacle_pos

    def step(self, move):
        self.reward = 0
        self.steps += 1  # illegal move as step
        if not helpers.get_poss_moves(self.board, team=0):
            self.reward += self.reward_loss
            self.score += self.reward
            return self.reward, True
        if self.Train:
            agent_move = move
        else:
            self.agents[0].living_pieces = self.living_pieces  # give agents pieces
            agent_move = self.agents[0].decide_move()
        # if not legal -> not change env, receive reward_illegal
        if not helpers.is_legal_move(self.board, agent_move):
            self.reward += self.reward_illegal
            self.illegal_moves += 1
            # print("illegal")
            self.score += self.reward
            done = self.goal_test()
            return self.reward, done  # environment does not change, agent should better choose only legal moves
        self.do_move(agent_move, team=0)

        # opponents move
        if self.opp_can_move:  # only if opponent is playing, killing his pieces wins
            if not helpers.get_poss_moves(self.board, team=1):
                self.reward += self.reward_win
                self.score = self.reward
                return self.reward, True
            self.agents[1].living_pieces = self.living_pieces  # give agents pieces
            opp_move = self.agents[1].decide_move()
            self.do_move(opp_move, team=1)  # assuming only legal moves selected

        done = self.goal_test()
        self.score += self.reward
        return self.reward, done

    def action_to_move(self, action, team):
        i = int(np.floor(action / 4))  # which piece: 0-3 is first 4-7 second etc.
        piece = self.living_pieces[team][i]  # TODO connect to environment
        piece_pos = piece.position  # where is the piece
        if piece_pos is None:
            move = (None, None)  # return illegal move
            return move
        action = action % 4  # 0-3 as direction
        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]  # a piece can move in four directions
        direction = moves[action]  # action: 0-3
        pos_to = [sum(x) for x in zip(piece_pos, direction)]  # go in this direction
        pos_to = tuple(pos_to)
        move = (piece_pos, pos_to)
        return move

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
        self.board[pos] = piece
        return

    def goal_test(self):
        for p in self.dead_pieces[1]:
            if p.type == 0:
                self.reward += self.reward_win
                # print("Red won, captured flag")
                return True
        for p in self.dead_pieces[0]:
            if p.type == 0:
                self.reward += self.reward_loss
                # print("Blue won, captured flag")
                return True
        if self.death_thresh is not None:
            if self.score < self.death_thresh:
                self.reward += self.reward_loss
                return True
        return False

    def show(self):
        fig = plt.figure(1)
        helpers.print_board(self.board)
        plt.title("Reward = {}".format(self.score))
        fig.canvas.draw()  # updates plot


################################################################################################################

class FindFlag(Env):
    def __init__(self, agent0, agent1):
        super(FindFlag, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10
        # self.death_thresh = -100

    def decide_pieces(self):
        known_place = []
        random_place = [pieces.Piece(3, 0, None), pieces.Piece(0, 1, None)]
        return known_place, random_place


class Maze(Env):
    def __init__(self, agent0, agent1):
        super(Maze, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step
        self.reward_illegal = -1
        self.reward_win = 10
        self.reward_iter = -1
        self.reward_loss = -1
        #
        # # TODO: Deprecate this
        # self.previous_pos = self.living_pieces[0][0]

    def decide_pieces(self):
        known_place = [pieces.Piece(0, 1, (4, 4))]
        random_place = [pieces.Piece(3, 0, None)]
        return known_place, random_place

    def decide_obstacles(self):
        return [(3, 1), (3, 2), (3, 3), (3, 4), (1, 0), (1, 1), (1, 2), (1, 3)]


class Survive(Env):
    def __init__(self, agent0, agent1):
        super(Survive, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step = -0.01
        self.reward_illegal = -0.1
        self.reward_win = 1
        self.reward_kill = 0.1
        self.reward_die = -0.1
        self.reward_loss = -1
        self.death_thresh = -100

    def decide_pieces(self):
        known_place = [pieces.Piece(0, 0, (0, 0)), pieces.Piece(0, 1, (4, 4)),
                       pieces.Piece(3, 0, (0, 1)), pieces.Piece(10, 0, (1, 0))]
        random_place = [pieces.Piece(3, 1, (4, 3)), pieces.Piece(10, 1, (3, 4)),
                        pieces.Piece(3, 1, (3, 3))]
        return known_place, random_place


class MiniMiniStratego(Env):
    def __init__(self, agent0, agent1):
        super(MiniMiniStratego, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10

    def decide_pieces(self):
        self.types_available = np.array([0, 1, 2, 2, 3, 10, 11])
        known_place = []
        # draw random setup for 10 figures for each team
        for team in (0, 1):
            setup_pos = [(i, j) for i in range(team * 3, 2 + team * 5) for j in range(5)]
            setup = np.random.choice(self.types_available, len(self.types_available), replace=False)
            for i, pos in enumerate(setup):
                known_place.append(pieces.Piece(setup[i], team, setup_pos[i]))
        random_place = []  # random_place is across whole board
        return known_place, random_place


class MiniStratego(Env):
    def __init__(self, agent0, agent1):
        super(MiniStratego, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10

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


def watch_game(env, step_time):
    """
    Watch two agents play against each other, step_time is
    """
    new_game = env
    env.Train = False
    done = False
    while not done:
        move = 0
        _, done = new_game.step(move)
        # print(env.reward)
        new_game.show()
        plt.pause(step_time)

    if env.reward > 0:  # reward_win is for Red
        outcome = "Red won!"
    else:
        outcome = "Blue won!"
    print(outcome)
    plt.title(outcome)
    plt.show(block=True)  # keep plot


    # State representation?
# max 7 channels
# own: movable,
#      immovable
# opp: known, movable
#             bomb
#      unknown, moved
#               not_moved
# obstacles
# own
# opp
# obstacles yes/no?
# own movable/immovable?
# opp movable/immovable?
# opp known/unknown?