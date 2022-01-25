from model_predictor import Hawkes
import numpy as np
import random
import sys
from utils import merge_timestamps, plot_single_point_obj_func, create_3d_random_walk, plot_two_points_obj_func


class LA(object):
    """This is to create L_RP-I Scheme LA class to act as the individual team member in an LA network that will represent
    the whole network, each LA will be assigned for a user in the network to conduct a random walk to learn about its
    authenticity and hidden influence to use for a mitigation of misinformation. The random walk is mathematically solving
    a constraint knapsack problem. The mitigation is obtained by learning to incentivize those hidden influencers
    with some amount from the budget"""

    def __init__(self, id, config, social_network, memory_depth):

        # a state here is just a discrete value that increasing it, means moving to another larger discrete state
        self.current_state = 0
        self.b = config.budget
        self.id = id  # id for the associated user
        self.social_network = social_network
        self.config = config
        self.state_updates = []  # stores the objective function x-axis values for plotting purposes
        self.initial_value = None
        self.obj_fun_v = None
        self.scores = []  # stores the objective function y-axis values for plotting purposes
        self.i = 0
        self.converged = False
        self.iters = [0]
        self.deltax = 0
        self.prev_state = 0
        self.memory_depth = memory_depth
        self.state_space = self.generate_state_space()
        self.move = 's'
        self.state_trans_moves_counts = np.array([(1, 1, 1) for _ in self.state_space], dtype=float)
        self.state_trans_moves_counts[0] = (0, 1, 1)
        self.state_trans_moves_counts[-1] = (1, 1, 0)
        self.state_trans_moves_reward_counts = np.array([(0, 0, 0) for _ in self.state_space], dtype=float)
        self.state_transition_matrix = self.initialize_matrix()
        self.feedback = None
        self.aimed_move = 's'
        self.states_stay_probs_trajectory = []
        self.prev_obj = None
        self.epsilon = config.epsilon

    def generate_state_space(self):
        """generates the LA state space for all possible states given a memory depth value parameter that represent the
        size of the state space"""

        return [i for i in np.linspace(0, self.b, num=self.memory_depth)]

    def step_right(self):
        """ for demo (no optimization) purposes, conduct random walk to the right (increase state)"""
        self.current_state += self.step

    def step_left(self):
        """for demo (no optimization) purposes, conduct random walk to the left (decrease state)"""

        if self.current_state - self.step >= 0:
            self.current_state -= self.step
        else:
            self.current_state = 0

    def explore_(self, controlled_mu, verbose):
        """a demo function to play the original problem without optimization to show the problem difficulty when
        stepping into the objective function and how noisy and non-stationary the function is"""

        self.step_right()
        self.deltax = self.current_state - self.state_updates[-1]
        mhp = Hawkes(None, self.social_network.norm_decay_factor, self.social_network.config)
        controlled_mu[self.id] += self.deltax
        pred_norm_time_stamps = mhp.simulate(controlled_mu, self.social_network.norm_A, verbose=False)
        _, merged_norm_timestamps = merge_timestamps({}, {},
                                                    self.social_network.normal_timestamps_before_simu, pred_norm_time_stamps)

        feedback, obj_fun_v = self.social_network.evaluate(self.id, merged_norm_timestamps,
                                                                      self.scores[-1], self.current_state, self.state_updates[-1])
        self.i += 1
        self.scores.append(obj_fun_v)
        self.state_updates.append(self.current_state)
        self.iters.append(self.i)

        if verbose:
            print(self.i, self.id, self.current_state, feedback)

        if self.i == 50:  # set an iteration number limit to exit exploring that user (LA)
            self.converged = True
            plot_single_point_obj_func(self.state_updates, self.scores, self.id)

        return controlled_mu

    def reward_func(self, feedback, knapsack_capacity):
        """reward function responsible for the final feedback from environment to the LA"""

        current_move = self.move
        current_knapsack_signal = knapsack_capacity/self.b >= .99
        current_state_signal = feedback
        if current_state_signal == 0 and current_knapsack_signal == 0 and current_move == 'r':
            return 0
        elif (current_state_signal == 1 or current_knapsack_signal == 1) and current_move == 'r':
            return 1
        elif current_state_signal == 1 and current_move == 'l':
            return 1
        elif current_state_signal == 0 and current_move == 'l':
            return 0
        elif current_knapsack_signal == 1 and current_move == 's':
            return 0
        else:
            return None

    def increase_state(self, current_state_index, consumed_states_value):
        """a method to increase the value of the current state, considered as moving to the right in
         a random walk manner"""

        if self.current_state != self.b:
            if consumed_states_value + self.deltax <= self.b:
                self.current_state = self.state_space[current_state_index + 1]
        else:
            self.current_state = self.b

    def decrease_state(self, current_state_index):
        """a method to decrease the value of the current state, considered as moving to the left
        in a random walk manner"""

        if self.current_state != 0:
            self.current_state = self.state_space[current_state_index - 1]
        else:
            self.current_state = 0

    def initialize_matrix(self):
        """state transition matrix initialize"""

        A = np.zeros((len(self.state_space), len(self.state_space)), dtype=float)
        for e, _ in enumerate(A):
            if e != 0 and e != self.state_space.index(self.b):
                A[e][e - 1] = 1 / 3
                A[e][e + 1] = 1 / 3
                A[e][e] = 1 / 3
            elif e == 0:
                A[e][e + 1] = .5
                A[e][e] = .5
            elif e == self.state_space.index(self.b):
                A[e][e - 1] = .5
                A[e][e] = .5
        return A

    def update_moves_probs(self, move_id, state_index):
        """updates state transition matrix given a reward function signal"""

        if self.current_state != 0 and self.current_state != self.b:
            if move_id == 'r':
                self.state_transition_matrix[state_index][state_index + 1] = self.state_trans_moves_reward_counts[state_index][2] / \
                                                                             self.state_trans_moves_counts[state_index][2]
                self.state_transition_matrix[state_index][state_index - 1] = (1 -
                                                                              self.state_transition_matrix[state_index][
                                                                                  state_index + 1]) / 2
                self.state_transition_matrix[state_index][state_index] = (1 - self.state_transition_matrix[state_index][
                    state_index + 1]) / 2
            elif move_id == 'l':
                self.state_transition_matrix[state_index][state_index - 1] = self.state_trans_moves_reward_counts[state_index][0] / \
                                                                             self.state_trans_moves_counts[state_index][0]
                self.state_transition_matrix[state_index][state_index + 1] = (1 -
                                                                              self.state_transition_matrix[state_index][
                                                                                  state_index - 1]) / 2
                self.state_transition_matrix[state_index][state_index] = (1 - self.state_transition_matrix[state_index][
                    state_index - 1]) / 2
            elif move_id == 's':
                self.state_transition_matrix[state_index][state_index] = self.state_trans_moves_reward_counts[state_index][1] / \
                                                                             self.state_trans_moves_counts[state_index][1]
                self.state_transition_matrix[state_index][state_index - 1] = (1 -
                                                                              self.state_transition_matrix[state_index][
                                                                                  state_index]) / 2
                self.state_transition_matrix[state_index][state_index + 1] = (1 -
                                                                              self.state_transition_matrix[state_index][
                                                                                  state_index]) / 2

        elif self.current_state == 0:
            if move_id == 'r':
                self.state_transition_matrix[state_index][state_index + 1] = self.state_trans_moves_reward_counts[state_index][2] / \
                                                                             self.state_trans_moves_counts[state_index][2]
                self.state_transition_matrix[state_index][state_index] = 1 - self.state_transition_matrix[state_index][
                    state_index + 1]
            elif move_id == 's':
                self.state_transition_matrix[state_index][state_index] = self.state_trans_moves_reward_counts[state_index][1] / \
                                                                             self.state_trans_moves_counts[state_index][1]
                self.state_transition_matrix[state_index][state_index + 1] = 1 - \
                                                                             self.state_transition_matrix[state_index][
                                                                                 state_index]
        elif self.current_state == self.b:
            if move_id == 'l':
                self.state_transition_matrix[state_index][state_index - 1] = self.state_trans_moves_reward_counts[state_index][0] / \
                                                                             self.state_trans_moves_counts[state_index][0]
                self.state_transition_matrix[state_index][state_index] = 1 - self.state_transition_matrix[state_index][
                    state_index - 1]
            elif move_id == 's':
                self.state_transition_matrix[state_index][state_index] = self.state_trans_moves_reward_counts[state_index][1] / \
                                                                             self.state_trans_moves_counts[state_index][1]
                self.state_transition_matrix[state_index][state_index - 1] = 1 - \
                                                                             self.state_transition_matrix[state_index][
                                                                                 state_index]

    def control(self, knapsack_capacity, controlled_mu, las, pred_norm_time_stamps_before_control):
        """a function that facilitates interaction as an LA with its environment"""
        state_value_to_try = self.get_next_transition()
        mhp = Hawkes(None, self.social_network.norm_decay_factor, self.social_network.config)
        if self.move == 'r':
            self.deltax = state_value_to_try - self.prev_state
            controlled_mu[self.id] += self.deltax
            knapsack_capacity += self.deltax
        elif self.move == 'l':
            self.deltax = self.prev_state - state_value_to_try
            if controlled_mu[self.id] - self.deltax >= 0 and knapsack_capacity - self.deltax >= 0:
                controlled_mu[self.id] -= self.deltax
                knapsack_capacity -= self.deltax
        else:
            self.deltax = self.prev_state - state_value_to_try

        if self.move != 's':
            if self.move == 'r':
                temp_current_state = self.state_space[self.state_space.index(self.state_updates[-1])+ 1]
            else:
                temp_current_state = self.state_space[self.state_space.index(self.state_updates[-1])- 1]
            pred_norm_time_stamps = mhp.simulate(controlled_mu, self.social_network.norm_A,
                                                 pred_norm_time_stamps_before_control,
                                                 verbose=self.config.verbose,
                                                 sampling = self.config.sampling, controlled_user_id=self.id)
            _, merged_norm_timestamps = merge_timestamps({}, {},
                                                         self.social_network.normal_timestamps_before_simu, pred_norm_time_stamps)

            feedback, self.obj_fun_v = self.social_network.evaluate(self.id, merged_norm_timestamps, self.scores[-1],
                                                                    self.state_updates[-1],
                                                                    temp_current_state, self.move)
        else:
            pred_norm_time_stamps = pred_norm_time_stamps_before_control
        state_index = self.state_space.index(self.current_state)
        if self.move != 's':
            final_feedback = self.reward_func(feedback, knapsack_capacity)
        else:
            final_feedback = None
        if final_feedback == 0 and self.move == 'r':
            self.aimed_move = self.move
            self.state_trans_moves_counts[state_index][2] += 1
            self.state_trans_moves_reward_counts[state_index][2] += 1
            self.update_moves_probs('r', state_index)
            self.increase_state(self.state_space.index(self.current_state), knapsack_capacity)
            self.feedback = 0
            self.move = 'r'

        elif final_feedback == 1 and self.move == 'r':

            self.aimed_move = self.move
            self.feedback = 1
            self.state_trans_moves_reward_counts[state_index][1] += 1
            self.state_trans_moves_counts[state_index][1] += 1
            self.update_moves_probs('s', state_index)
            self.move = 's'
            controlled_mu[self.id] -= self.deltax
            knapsack_capacity -= self.deltax

        elif final_feedback == 1 and self.move == 'l':
            self.aimed_move = self.move
            self.state_trans_moves_reward_counts[state_index][1] += 1
            self.state_trans_moves_counts[state_index][1] += 1
            self.update_moves_probs('s', state_index)
            self.move = 's'
            self.feedback = 1
            if controlled_mu[self.id] >= 0 and knapsack_capacity >= 0:
                controlled_mu[self.id] += self.deltax
                knapsack_capacity += self.deltax

        elif final_feedback == 0 and self.move == 'l':
            self.aimed_move = self.move
            self.state_trans_moves_counts[state_index][0] += 1
            self.state_trans_moves_reward_counts[state_index][0] += 1
            self.update_moves_probs('l', state_index)
            self.feedback = 0
            self.move = 'l'
            self.decrease_state(self.state_space.index(self.current_state))
        elif self.move == 's':
            self.aimed_move = self.move
            self.move = 's'
            self.feedback = None

        self.prev_state = self.current_state
        self.scores.append(self.obj_fun_v)
        self.state_updates.append(self.current_state)
        new_state_index = self.state_space.index(self.current_state)
        self.check_converged(new_state_index, las)

        # check for standstill states
        trapped, trapp_indx = self.is_trapped_state(new_state_index)
        if trapped:
            if self.explore(knapsack_capacity):
                self.aimed_move = self.move
                self.state_trans_moves_counts[state_index][2] += 1
                self.state_trans_moves_reward_counts[state_index][2] += 1
                self.update_moves_probs('r', state_index)
                self.feedback = 0
                self.move = 'r'
                self.increase_state(self.state_space.index(self.current_state), knapsack_capacity)
            else:
                self.state_trans_moves_counts[new_state_index][1] += 1
                self.state_trans_moves_reward_counts[new_state_index][1] += 1
                self.update_moves_probs('s', new_state_index)

        return knapsack_capacity, controlled_mu, pred_norm_time_stamps

    def get_next_transition(self):
        """get the value of either increase, decrease, or neutral to the current la state, these values are
         decided according to state transition probabilities from the current state in the state
         transition matrix
         """

        state_index = self.state_space.index(self.current_state)

        if self.current_state != 0 and self.current_state != self.b:
            self.move = self.w_choice([('l', self.state_transition_matrix[state_index][state_index - 1] * 100),
                                       ('s', self.state_transition_matrix[state_index][state_index] * 100),
                                       ('r', self.state_transition_matrix[state_index][state_index + 1] * 100)])
        elif self.current_state == 0:
            self.move = self.w_choice([('l', 0), ('s', self.state_transition_matrix[state_index][state_index] * 100),
                                       ('r', self.state_transition_matrix[state_index][state_index + 1] * 100)])
        elif self.current_state == self.b:
            self.move = self.w_choice([('l', self.state_transition_matrix[state_index][state_index - 1] * 100),
                                       ('s', self.state_transition_matrix[state_index][state_index] * 100), ('r', 0)])

        if self.move == 'l':
            return self.state_space[state_index - 1]
        elif self.move == 's':
            return self.state_space[state_index]
        else:
            return self.state_space[state_index + 1]

    def w_choice(self, seq):
        """helper function to randomly select next transition"""

        total_prob = sum(item[1] for item in seq)
        chosen = random.uniform(0, total_prob)
        cumulative = 0
        for item, probality in seq:
            cumulative += probality
            if cumulative > chosen:
                return item

    def check_converged(self, state_index, las):
        """check if an LA converged or not and if so, it plots the objective function trajectory and joint random walks
         examples"""

        if len(self.state_updates) > 4:
            if (self.state_updates[-1] == self.state_updates[-2] == \
                    self.state_updates[-3] == self.state_updates[-4] ==\
                    self.state_updates[-5]) and self.state_transition_matrix[state_index][state_index] >= self.config.threshold:
                self.converged = True
                if self.config.graphs:
                    i_sorted_x = sorted(list(set(self.state_updates)))
                    i_sorted_y = [self.scores[list(i_sorted_x).index(e)] for e in i_sorted_x]
                    plot_single_point_obj_func(self.state_updates, self.scores, self.id, self.current_state)

                    joint_las = []
                    joint_las.append((self.id, self.state_updates))
                    for la in las:
                        if len(joint_las) >= 3:
                            create_3d_random_walk(joint_las[-3][0], joint_las[-2][0], joint_las[-1][0], joint_las[-3][1], joint_las[-2][1], joint_las[-1][1])
                        if la.id != self.id:
                            if la.converged:
                                joint_las.append((la.id, la.state_updates))
                                j_sorted_x = sorted(list(set(la.state_updates)))
                                j_sorted_y = [la.scores[list(j_sorted_x).index(e)] for e in j_sorted_x]
                                plot_two_points_obj_func(self.id, la.id, i_sorted_x, j_sorted_x, i_sorted_y, j_sorted_y, True)
                                plot_two_points_obj_func(self.id, la.id, i_sorted_x, j_sorted_x, i_sorted_y, j_sorted_y, False)

            else:
                self.converged = False
        else:
            self.converged = False

    def is_trapped_state(self, state_index):
        """helper function to check if there is a trapped state where increasing it or decreasing it having very high
        probs of transitions, which causing a two-states trapped loop, the function searchs if the current LA state
         is a member of such two-states trapping loop"""

        if (self.state_transition_matrix[state_index][state_index + 1] and \
        self.state_transition_matrix[state_index + 1][state_index] >= self.config.threshold):
            return True, 0
        elif (self.state_transition_matrix[state_index][state_index-1]
        and self.state_transition_matrix[state_index-1][state_index] >= self.config.threshold):
            return True, 1
        else:
            return False, None

    def explore(self, knapsack_capacity):
        """a function that determines how a forced-exploration of new states is made"""

        r = random.random()
        delta_x_explored = self.state_space[self.state_space.index(self.current_state)+1] - self.current_state

        # we set force-explor to zero probability in our default experiments. Hence, we only follow the
        # state transition matrix for the stochastic explor/ exploit
        if r <= self.epsilon and knapsack_capacity + delta_x_explored <= self.b:
            return True
        else:
            return False



