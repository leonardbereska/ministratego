"""
Agent decides the initial setup and decides which action to take
"""
import random
import numpy as np
import pieces
import copy
from collections import Counter
from scipy import spatial
from scipy import optimize
import battleMatrix
import helpers


class Agent:
    def __init__(self, team, setup):
        self.team = team
        self.other_team = (self.team + 1) % 2
        self.setup = setup
        self.board = np.empty((5, 5), dtype=object)
        for idx, piece in np.ndenumerate(setup):
            piece.hidden = False
            self.board[piece.position] = piece

        self.move_count = 0

        self.last_N_moves = []
        self.pieces_last_N_Moves_beforePos = []
        self.pieces_last_N_Moves_afterPos = []

        obstacle = pieces.Piece(99, 99, (2, 2))
        obstacle.hidden = False
        self.board[2, 2] = obstacle  # set obstacle

        self.battleMatrix = battleMatrix.get_battle_matrix()

        # fallen pieces bookkeeping
        self.deadPieces = []
        dead_piecesdict = dict()
        self.types_available = [0, 1, 2, 2, 2, 3, 3, 10, 11, 11]
        for type_ in set(self.types_available):
            dead_piecesdict[type_] = 0
        self.deadPieces.append(dead_piecesdict)
        self.deadPieces.append(copy.deepcopy(dead_piecesdict))

        self.ordered_opp_pieces = []

    def install_opp_setup(self, opp_setup):
        self.assignment_dict = dict()
        enemy_types = [piece.type for idx, piece in np.ndenumerate(opp_setup)]
        for idx, piece in np.ndenumerate(opp_setup):
            piece.potential_types = copy.copy(enemy_types)
            self.ordered_opp_pieces.append(piece)
            piece.hidden = True
            self.board[piece.position] = piece

    def update_board(self, updated_piece, board=None):
        if board is None:
            board = self.board
        if updated_piece[1] is not None:
            updated_piece[1].change_position(updated_piece[0])
        board[updated_piece[0]] = updated_piece[1]
        return board

    def decide_move(self):
        """
        Agent has to implement which action to decide on given the state
        """
        raise NotImplementedError

    def do_move(self, move, board=None, bookkeeping=True, true_gameplay=False):
        """
        :param move: tuple or array consisting of coordinates 'from' at 0 and 'to' at 1
        """
        from_ = move[0]
        to_ = move[1]
        turn = self.move_count % 2
        fight_outcome = None
        if board is None:
            board = self.board
            board[from_].has_moved = True
        moving_piece = board[from_]
        attacked_field = board[to_]
        self.last_N_moves.append(move)
        self.pieces_last_N_Moves_afterPos.append(attacked_field)
        self.pieces_last_N_Moves_beforePos.append(moving_piece)
        if not board[to_] is None:  # Target field is not empty, then has to fight
            if board is None:
                # only uncover them when the real board is being played on
                attacked_field.hidden = False
                moving_piece.hidden = False
            fight_outcome = self.fight(moving_piece, attacked_field, collect_dead_pieces=bookkeeping)
            if fight_outcome is None:
                print('Warning, cant let pieces of same team fight!')
                return False
            elif fight_outcome == 1:
                self.update_board((to_, moving_piece), board=board)
                self.update_board((from_, None), board=board)
            elif fight_outcome == 0:
                self.update_board((to_, None), board=board)
                self.update_board((from_, None), board=board)
            else:
                self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.team:
                    self.update_prob_by_fight(attacked_field)
                else:
                    self.update_prob_by_fight(moving_piece)
        else:
            self.update_board((to_, moving_piece), board=board)
            self.update_board((from_, None), board=board)
            if true_gameplay:
                if turn == self.other_team:
                    self.update_prob_by_move(move, moving_piece)
        return board, fight_outcome

    def fight(self, piece_att, piece_def, collect_dead_pieces=True):
        """
        Determine the outcome of a fight between two pieces: 1: win, 0: tie, -1: loss
        add dead pieces to deadFigures
        """
        outcome = self.battleMatrix[piece_att.type, piece_def.type]
        if collect_dead_pieces:
            if outcome == 1:
                piece_def.dead = True
                self.deadPieces[piece_def.team][piece_def.type] += 1
            elif outcome == 0:
                self.deadPieces[piece_def.team][piece_def.type] += 1
                piece_def.dead = True
                self.deadPieces[piece_att.team][piece_att.type] += 1
                piece_att.dead = True
            elif outcome == -1:
                self.deadPieces[piece_att.team][piece_att.type] += 1
                piece_att.dead = True
            if piece_att.guessed or piece_def.guessed:
                outcome *= 2
            return outcome
        if piece_att.guessed or piece_def.guessed:
            outcome *= 2
        return outcome

    def update_prob_by_fight(self, *args):
        pass

    def update_prob_by_move(self, *args):
        pass

    def get_poss_actions(self, board, team):  # TODO: change references to helper's version
        return helpers.get_poss_actions(board, team)

    def is_legal_move(self, move_to_check, board):  # TODO: change references to helper's version
        return helpers.is_legal_move(board, move_to_check)

    def analyze_board(self):
        pass


class RandomAgent(Agent):
    """
    Agent who chooses his initial setup and actions at random
    """
    def __init__(self, team, setup):
        super(RandomAgent, self).__init__(team=team, setup=setup)

    def decide_move(self):
        actions = helpers.get_poss_actions(self.board, self.team)
        # ignore state, do random action
        action = random.choice(actions)
        return action


class SmartSetup(Agent):
    """
    RandomAgent with smart initial setup
    """
    def __init__(self, team, setup):
        super(SmartSetup, self).__init__(team=team)
        self.setup = setup

    def decide_move(self):
        actions = helpers.get_poss_actions(self.board, self.team)
        # ignore state, do random action
        action = random.choice(actions)
        return action


class ExpectiSmart(Agent):
    def __init__(self, team, setup):
        super(ExpectiSmart, self).__init__(team=team, setup=setup)

        self.kill_reward = 10
        self.neutral_fight = 2
        self.winGameReward = 100
        self.certainty_multiplier = 1.2

        self.battleMatrix = battleMatrix.get_battle_matrix()

    def decide_move(self):
        return self.minimax(max_depth=6)

    def minimax(self, max_depth):
        curr_board = copy.deepcopy(self.board)
        curr_board = self.draw_consistent_enemy_setup(curr_board)
        chosen_action = self.max_val(curr_board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action

    def max_val(self, board, current_reward, alpha, beta, depth):
        # this is what the expectimax agent will think

        my_doable_actions = helpers.get_poss_actions(board, self.team)

        # check for end-state scenario
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = -float('inf')
        best_action = None
        for action in my_doable_actions:
            board = self.do_move(action, board=board,  bookkeeping=False, true_gameplay=False)
            fight_result = board[1]
            board = board[0]
            temp_reward = current_reward
            if fight_result is not None:
                if fight_result == 1:
                    temp_reward += self.kill_reward
                elif fight_result == 2:
                    temp_reward += int(self.certainty_multiplier*self.kill_reward)
                elif fight_result == 0:
                    temp_reward += self.neutral_fight  # both pieces die
                elif fight_result == -1:
                    temp_reward += -self.kill_reward
                elif fight_result == -2:
                    temp_reward += -int(self.certainty_multiplier * self.kill_reward)
            new_val = self.min_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val < new_val:
                val = new_val
                best_action = action
            if val >= beta:
                board = self.undo_last_move(board)
                best_action = action
                return val, best_action
            alpha = max(alpha, val)
            board = self.undo_last_move(board)
        return val, best_action

    def min_val(self, board, current_reward, alpha, beta, depth):
        # this is what the opponent will think, the min-player

        my_doable_actions = helpers.get_poss_actions(board, self.other_team)
        # check for end-state scenario first
        goal_check = self.goal_test(my_doable_actions, board)
        if goal_check or depth == 0:
            if goal_check == True:  # Needs to be this form, as -100 is also True for if statement
                return current_reward, (None, None)
            return current_reward + goal_check, (None, None)

        val = float('inf')  # inital value set, so min comparison later possible
        best_action = None
        for action in my_doable_actions:
            board = self.do_move(action, board=board, bookkeeping=False, true_gameplay=False)
            fight_result = board[1]
            board = board[0]
            temp_reward = current_reward
            if fight_result is not None:
                if fight_result == 1:
                    temp_reward += -self.kill_reward
                elif fight_result == 2:
                    temp_reward += -int(self.certainty_multiplier*self.kill_reward)
                elif fight_result == 0:
                    temp_reward += self.neutral_fight  # both pieces die
                elif fight_result == -1:
                    temp_reward += self.kill_reward
                elif fight_result == -2:
                    temp_reward += int(self.certainty_multiplier * self.kill_reward)
            new_val = self.max_val(board, temp_reward, alpha, beta, depth-1)[0]
            if val > new_val:
                val = new_val
                best_action = action
            if val <= alpha:
                board = self.undo_last_move(board)
                return val, best_action
            beta = min(beta, val)
            board = self.undo_last_move(board)
        return val, best_action

    def goal_test(self, actions_possible, board=None):
        if board is not None:
            flag_alive = [False, False]
            for pos, piece in np.ndenumerate(board):
                if piece is not None and piece.type == 0:
                    flag_alive[piece.team] = True
            if not flag_alive[self.other_team]:
                return self.winGameReward
            if not flag_alive[self.team]:
                return -self.winGameReward
        else:
            if 0 in self.deadPieces[0] or 0 in self.deadPieces[1]:
                # print('flag captured')
                return True
        if not actions_possible:
            # print('cannot move anymore')
            return True
        else:
            return False

    def update_prob_by_fight(self, enemy_piece):
        enemy_piece.potential_types = [enemy_piece.type]

    def update_prob_by_move(self, move, moving_piece):
        move_dist = spatial.distance.cityblock(move[0], move[1])
        if move_dist > 1:
            moving_piece.hidden = False
            moving_piece.potential_types = moving_piece.type
        else:
            immobile_enemy_types = [idx for idx, type in enumerate(moving_piece.potential_types)
                                    if type in [0, 11]]
            moving_piece.potential_types = np.delete(moving_piece.potential_types, immobile_enemy_types)



    def draw_consistent_enemy_setup(self, board):
        # get information about enemy pieces (how many, which alive, which types, and indices in assign. array)
        enemy_pieces = copy.deepcopy(self.ordered_opp_pieces)
        enemy_pieces_alive = [piece for piece in enemy_pieces if not piece.dead]
        types_alive = [piece.type for piece in enemy_pieces_alive]

        # do the following as long as the drawn assignment is not consistent with the current knowledge about them
        consistent = False
        sample = None
        while not consistent:
            # choose len(types_alive) many pieces randomly
            sample = np.random.choice(types_alive, len(types_alive), replace=False)
            consistent = True
            for idx, piece in enumerate(enemy_pieces_alive):
                if sample[idx] not in piece.potential_types:
                    consistent = False
        for idx, piece in enumerate(enemy_pieces_alive):
            piece.guessed = not piece.hidden
            piece.type = sample[idx]
            if piece.type in [0, 11]:
                piece.can_move = False
                piece.move_radius = 0
            elif piece.type == 2:
                piece.can_move = True
                piece.move_radius = float('inf')
            else:
                piece.can_move = True
                piece.move_radius = 1
            piece.hidden = False
            board[piece.position] = piece
        return board

    def undo_last_move(self, board):
        last_move = self.last_N_moves.pop()
        if last_move is None:
            raise ValueError("No last move to undo detected!")
        before_piece = self.pieces_last_N_Moves_beforePos.pop()
        board[last_move[0]] = before_piece
        # the piece at the 'before' position was the one that moved, so needs its
        # last entry in the move history deleted
        before_piece.position = last_move[0]
        #before_piece.positions_history.pop()
        board[last_move[1]] = self.pieces_last_N_Moves_afterPos.pop()
        return board


class OmniscientExpectiSmart(ExpectiSmart):
    def __init__(self, team, setup=None):
        super(OmniscientExpectiSmart, self).__init__(team=team, setup=setup)
        self.setup = setup
        self.winFightReward = 10
        self.neutralFightReward = 5
        self.winGameReward = 1000

    def install_opp_setup(self, opp_setup):
        super().install_opp_setup(opp_setup)
        self.unhide_all()

    def unhide_all(self):
        for pos, piece in np.ndenumerate(self.board):
            if piece is not None:
                piece.hidden = False

    def minimax(self, max_depth):
        chosen_action = self.max_val(self.board, 0, -float("inf"), float("inf"), max_depth)[1]
        return chosen_action


class Reinforce(Agent):
    """
    Agent approximating action-value functions with an artificial neural network
    trained with Q-learning
    """
    def __init__(self, team):
        super(Reinforce, self).__init__(team=team)
        self.BATCH_SIZE = 128  # 128
        self.GAMMA = 0.99
        self.EPS_START = 0.02
        self.EPS_END = 0.001
        self.EPS_DECAY = 100
        self.N_SMOOTH = 10  # plotting scores averaged over this number of episodes
        self.EVAL = False  # evaluation mode: controls verbosity of output e.g. printing non-optimal moves
        self.VERBOSE = 1  # level of printed output verbosity

        self.num_episodes = 10000  # training for how many episodes

        state_dim = env.get_state().shape[1]  # state has state_dim*5*5 values (board_size * depth of representation)
        self.model = models.Finder(state_dim)
        self.model.load_state_dict(torch.load('./saved_models/finder.pkl'))
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = helpers.ReplayMemory(1000)

    def decide_move(self):
        action = self.select_action(state, p_random=0.1)
        # action -> to train function
        move = self.action_to_move(action, self.team)
        return move

    def user_action(self):
        direction = input("Type direction\n")
        keys = ('w', 's', 'a', 'd', 'i', 'k', 'j', 'l')
        if direction not in keys:
            direction = input("Try typing again\n")
        return keys.index(direction)

    def select_action(self, state, p_random):
        """
        Agents action is one of four directions
        :return: action 0: up, 1: down, 2: left, 3: right (cross in prayer)
        """
        sample = random.random()
        if sample > p_random:
            # deterministic action selection
            # output = model(Variable(state, volatile=True)).data
            # # print(output.numpy())
            # action = output.max(1)[1].view(1, 1)  # choose maximum index
            # return action

            # probabilistic action selection, network outputs state-action values in (0, 1)
            state_action_values = self.model(Variable(state, volatile=True))
            p = list(state_action_values.data[0].numpy())
            p = [int(p_i * 1000) / 1000 for p_i in p]
            p[3] = 1 - sum(p[0:3])  # artificially make probs sum to one
            # print(p)  # print probabilities
            action = np.random.choice(np.arange(0, 4), p=p)
            action = int(action)  # normal int not numpy int
            return torch.LongTensor([[action]])
        else:
            return torch.LongTensor([[random.randint(0, 3)]])

    def action_to_move(self, action, team):
        i = int(np.floor(action / 4))  # which piece: 0-3 is first 4-7 second etc.
        piece = env.living_pieces[team][i]
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

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return  # not optimizing for not enough memory
        transitions = self.memory.sample(self.BATCH_SIZE)  # sample memories batch
        batch = helpers.Transition(*zip(*transitions))  # transpose the batch

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(torch.FloatTensor))  # zero for teminal states
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[
            0]  # what would the model predict for next
        next_state_values.volatile = False  # requires_grad = False to not mess with loss
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch  # compute the expected Q values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)  # compute Huber loss

        # optimize network
        self.optimizer.zero_grad()  # optimize towards expected q-values
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, env, num_episodes):
        episode_scores = []  # score = total reward
        for i_episode in range(num_episodes):
            env.reset()  # initialize environment
            state = env.get_state()  # initialize state
            while True:
                # act in environment
                p_random = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
                action = env.agents[0].select_action(state, p_random)  # random action with probability p_random
                reward_value, done = env.step()  # environment step for action
                if self.VERBOSE > 1:
                    print(action[0, 0] + 1, reward_value)
                reward = torch.FloatTensor([reward_value])

                # save transition as memory and optimize model
                if done:  # if terminal state
                    next_state = None
                else:
                    next_state = env.get_state()
                self.memory.push(state, action, next_state, reward)  # store the transition in memory
                state = next_state  # move to the next state
                sefl.optimize_model()  # one step of optimization of target network

                if done:
                    print("Episode {}/{}".format(i_episode, num_episodes))
                    print("Score: {}".format(env.score))
                    print("Noise: {}".format(p_random))
                    print("Illegal: {}/{}\n".format(env.illegal_moves, env.steps))
                    episode_scores.append(env.score)
                    if self.VERBOSE > 1:
                        helpers.plot_scores(episode_scores)  # takes run time
                    break
            if i_episode % 100 == 2:
                if self.VERBOSE > 1:
                    self.run_env(env, False, 1)

    def run_env(self, env, user_test, n_runs=100):
        global EVAL
        EVAL = True  # switch evaluation mode on
        for i in range(n_runs):
            env.reset()
            env.show()
            done = False
            while not done:
                state = env.get_state()
                if user_test:
                    action = self.user_action()
                else:
                    action = self.select_action(state, 0.00)
                    action = action[0, 0]
                _, done = env.step(action)
                env.show()
                if done and env.reward == env.reward_win:
                    print("Won!")
                elif (done and env.reward == env.reward_loss) or env.score < -5:
                    print("Lost")
                    break
