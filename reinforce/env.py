from matplotlib import pyplot as plt
import agent
import numpy as np
import copy as cp
import random
import torch
import pieces
import helpers
import battleMatrix
import agent
import six


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

        # place fixed position pieces
        for piece in known_pieces:
            self.board[piece.position] = piece
            positions.remove(piece.position)
            self.living_pieces[piece.team].append(piece)
        # place random position pieces
        c = list(np.random.choice(len(positions), len(random_pieces), replace=False))
        for p in random_pieces:
            self.board[positions[c.pop()]] = p
            self.living_pieces[p.team].append(p)

        # give agents board
        self.agents = (agent0, agent1)
        agent0.board = cp.deepcopy(self.board)
        agent1.board = cp.deepcopy(self.board)

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

    def get_state(self):
        raise NotImplementedError

    def step(self):
        self.reward = 0
        self.steps += 1  # illegal move as step

        # agent_move = self.action_to_move(action, team=0)

        # are actions possible? -> if not: lose
        if not helpers.get_poss_actions(self.board, team=0):
            self.reward += self.reward_loss
            self.score += self.reward
            return self.reward, True
        # state = self.get_state()
        agent_move = self.agents[0].decide_move()
        # if not legal -> not change env, receive reward_illegal
        if not helpers.is_legal_move(self.board, agent_move):
            self.reward += self.reward_illegal
            self.illegal_moves += 1
            self.score += self.reward
            done = self.goal_test()
            return self.reward, done  # environment does not change, agent should better choose only legal moves
        self.do_move(agent_move, team=0)

        # opponents move
        if self.opp_can_move:  # only if opponent is playing, killing his pieces wins
            if not helpers.get_poss_actions(self.board, team=1):
                self.reward += self.reward_win
                self.score = self.reward
                return self.reward, True
            opp_move = self.agents[1].decide_move()
            self.do_move(opp_move, team=1)  # assuming only legal moves selected

        done = self.goal_test()
        self.score += self.reward
        return self.reward, done

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


########################################################

class FindFlag(Env):
    def __init__(self, agent0, agent1):
        super(FindFlag, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_step = -0.01
        self.reward_illegal = -1
        self.reward_win = 1
        self.death_thresh = -100

    def decide_pieces(self):
        known_place = []
        random_place = [pieces.Piece(3, 0, None), pieces.Piece(0, 1, None)]
        return known_place, random_place

    # def decide_obstacles(self):
    #     return []

    def get_state(self):
        state_dim = 2
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:  # opponents team
                    board_state[tuple([1] + list(pos))] = 1  # flag
                    # else:
                    #     board_state[tuple([2] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Escape(Env):
    def __init__(self, agent0, agent1):
        super(Escape, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_illegal = -1
        self.reward_win = 100
        self.reward_loss = -10
        self.reward_iter = -1

    def decide_pieces(self):
        known_place = []
        random_place = [pieces.Piece(3, 0, None), pieces.Piece(10, 1, None),  # pieces.Piece(10, 1, None),
                        pieces.Piece(0, 1, None)]
        return known_place, random_place

    def get_state(self):
        state_dim = 3
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:  # opponents team
                    if p.type == 0:
                        board_state[tuple([1] + list(pos))] = 1  # flag
                    else:
                        board_state[tuple([2] + list(pos))] = 1  # opp piece
                        # else:
                        #     board_state[tuple([3] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state


class Maze(Env):
    def __init__(self, agent0, agent1):
        super(Maze, self).__init__(agent0=agent0, agent1=agent1)
        # self.reward_step
        self.reward_illegal = -1
        self.reward_win = 10
        self.reward_iter = -1
        self.reward_loss = -1

        # TODO: Deprecate this
        self.previous_pos = self.living_pieces[0][0]

    def decide_pieces(self):
        known_place = [pieces.Piece(0, 1, (4, 4))]
        random_place = [pieces.Piece(3, 0, None)]
        return known_place, random_place

    def decide_obstacles(self):
        return [(3, 1), (3, 2), (3, 3), (3, 4), (1, 0), (1, 1), (1, 2), (1, 3)]

    def get_state(self):
        state_dim = 3
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = 1  # represent type
                elif p.team == 1:
                    board_state[tuple([1] + list(pos))] = 1  # flag
                else:
                    board_state[tuple([2] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state

        # def get_state(self):
        #     for pos in self.board_positions:
        #         p = self.board[pos]
        #         if p is not None:
        #             if p.team == 0:
        #                 board_state = pos
        #     return torch.FloatTensor(board_state)


class Kill(Env):
    def __init__(self, agent0, agent1):
        super(Kill, self).__init__(agent0=agent0, agent1=agent1)
        self.reward_step = -0.1
        self.reward_illegal = -1
        self.reward_win = 10
        self.reward_kill = 1

    def decide_pieces(self):
        random_place = [pieces.Piece(10, 0, None), pieces.Piece(3, 1, None), pieces.Piece(3, 1, None)]
        return [], random_place

    def get_state(self):
        state_dim = 3
        board_state = np.zeros((state_dim, 5, 5))  # zeros for empty field
        for pos in self.board_positions:
            p = self.board[pos]
            if p is not None:  # piece on this field
                if p.team == 0:  # agents team
                    board_state[tuple([0] + list(pos))] = p.type  # represent type
                elif p.team == 1:
                    board_state[tuple([1] + list(pos))] = 1  # other team
                else:
                    board_state[tuple([2] + list(pos))] = 1  # obstacle

        board_state = torch.FloatTensor(board_state)
        board_state = board_state.view(1, state_dim, 5, 5)  # add dimension for more batches
        return board_state

    # class SmallGame(Env):
    #     def __init__(self):
    #         super(SmallGame, self).__init__()
    #         self.reward_step = -0.1
    #         self.reward_illegal = -1
    #         self.reward_win = 10
    #
    #     def decide_pieces(self):
    #         pieces_list = [pieces.Piece(3, 0), pieces.Piece(0, 0), pieces.Piece(3, 1), pieces.Piece(0, 1)]
    #         return pieces_list


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


def watch_game(env, step_time):
    """
    Watch two agents play against each other, step_time is
    """
    new_game = env
    done = False
    while not done:
        _, done = new_game.step()
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

# for testing the ministratego!
good_setup2 = np.empty((2, 5), dtype=int)
good_setup2[0, 0] = 3
good_setup2[0, 1] = 11
good_setup2[0, 2] = 0
good_setup2[0, 3] = 11
good_setup2[0, 4] = 1
good_setup2[1, 0] = 2
good_setup2[1, 1] = 2
good_setup2[1, 2] = 10
good_setup2[1, 3] = 2
good_setup2[1, 4] = 3

rd_setup = np.empty((5, 5), )
rd_setup[:, :] = None
rd_setup[0, 1] = 0
rd_setup[1, 0] = 11
rd_setup[2, 3] = 10
rd_setup[2, 1] = 10
rd_setup[1, 2] = 10
rd_setup[0, 4] = 10

setup_agent0 = np.empty((2, 5), dtype=object)
#setup_agent1 = np.empty((2, 5), dtype=object)
setup_agent1 = []
for pos in ((i, j) for i in range(2) for j in range(5)):
    setup_agent0[pos] = pieces.Piece(good_setup2[pos], 0, (4-pos[0], 4-pos[1]))
for pos, type in np.ndenumerate(rd_setup):
    if not type != type:  # check if type is NaN
        setup_agent1.append(pieces.Piece(int(type), 1, pos))
env = MiniStratego(agent.OmniscientExpectiSmart(0, setup_agent0), agent.RandomAgent(1, setup_agent1))
while True:
    watch_game(env, 0.001)
