#################################################################
#   this module is used from (Abouzeid. et. al 2021) to use as  #
#   an comparative method to our work                           #
#################################################################

import time
import numpy as np
import random
import math
from tick.hawkes import HawkesADM4, SimuHawkesExpKernels
from tqdm import tqdm
from matplotlib import pyplot as plt
from Comparable_Network_Measures import *


class LA_RAND_WALK(object):
    """This is to create L_RP-I Scheme LA class to act as the individual team member in an LA network that will represent
    the whole network, each LA will be assigned for a user in the network"""

    def __init__(self, lambdaR, budget, id):
        """object initializer method

        Parameters
        ------------------
        lambdaR: float
                the LA action selection probability update factor
        state_limit: float
                the final allowed state that LA can go far to, that value is considered from the optimization
                constraint (allowed budget for incentivization)
        id: int
                an id assigned when creating the LA object to map each LA to a user id from the Hawkes module data"""

        self.LambdaR = lambdaR
        # a state here is just a continues value that increasing it, means moving to another state
        self.current_state = 0
        self.expected_values = []
        self.budget = budget
        # to track how each user (LA) incentives were done
        self.states_trajectory = [self.current_state]
        # the LA has only one action (incentivize), and this is the action selection probability
        self.good_candidate_prob = .5
        self.bad_candidate_prob = 1 - self.good_candidate_prob
        self.network = None
        self.inc_amount =  0
        self.id = id
        self.exploration_count = 0
        self.rewarded_count = 0
        self.penalized_count = 0
        self.step_value = .01
        self.current_action = None

    def update_la_probs(self, signal):
        if signal == 1:
            self.rewarded_count\
                += 1
            self.good_candidate_prob = self.rewarded_count / self.exploration_count
            self.bad_candidate_prob = 1 - self.good_candidate_prob
        else:
            self.penalized_count += 1
            self.bad_candidate_prob = self.penalized_count / self.exploration_count
            self.good_candidate_prob = 1 - self.bad_candidate_prob

    def increase_state(self, current_budget, all_budget):
        """a method to accept the act of incentivization for the given user (LA), it is called upon successful evaluation
         of the current la (user) incentivization"""
        self.current_state += self.inc_amount

    def decrease_state(self):
        """a method to deny the act of incentivization for the given user (LA), it is called upon failure evaluation
         of the current la (user) incentivization"""
        if self.current_state - self.inc_amount >= 0:
            self.current_state -= self.inc_amount
        else:
            self.current_state = 0


class LATeam(object):
    """this class is for the LA team to be sampled for the network, each LA team will have its individual LAs and the new sampled
    (initiated LATeam class) can have LAs from an old teams in old iterations samples and so on."""

    def __init__(self, network_LAs, team_size, realizations_n, realizations_bounds, end_time, la_m_depth, sensetivity_eval_param):
        """initiating the LATeam class with required parameters to form the team

        Parameters
        -------------------
         network_LAs: list of LA objects
                    list of current LA objects with their current states like probabilities of incentivization (action selection)
         team_size: int
                    number of la to be evaluated per single iteration
         realizations: int
                    number of time stages
        """
        self.team_size = team_size
        self.network_LAs = network_LAs
        self.team_members_values = []
        self.realizations_n = realizations_n
        self.realizations_bounds = realizations_bounds

        self.end_time = end_time
        self.epsilon = .1
        self.current_explored_LAs = []
        self.current_replaced_LAs = []
        self.explore_decay_factor = 0
        self.current_to_exploit_las_ids = []
        self.la_m_depth = la_m_depth
        self.sensetivity_eval_param = sensetivity_eval_param

    def sampleLATeam(self, scheme, x, budget):
        """this function returns a list of LAs to evaluate the mitigation with them, at each iteration of the
        network, a team is sampled, hopefully till convergence of those LAs that stays in any sampled new team

        Parameters
        --------------------
        i: int
            current iteration number, used to check if the number is odd or even so sample size is extended by one if even
             for evaluating and exploring new element(s)"""

        self.current_explored_LAs = []
        self.team_members_values = []
        self.current_replaced_LAs = []
        if x == 0:
            random.shuffle(self.network_LAs)
        rand_la = self.network_LAs[x]
        to_exploit_las = [(rand_la.id, round(rand_la.good_candidate_prob))]
        self.current_to_exploit_las_ids.append(rand_la.id)
        self.team_members_values.append((rand_la.id, rand_la.good_candidate_prob))
        if scheme in [0, 1, 2]:
            rand_la.exploration_count += 1
        elif scheme == 1 or scheme == 2:
            if rand_la.good_candidate_prob > rand_la.bad_candidate_prob:
                rand_la.current_action = 1
                rand_la.good_candidate_exploration_count += 1
            elif rand_la.good_candidate_prob < rand_la.bad_candidate_prob:
                rand_la.current_action = 0
                rand_la.bad_candidate_exploration_count += 1
            else:
                if random.random() > .5:
                    rand_la.current_action = 1
                    rand_la.good_candidate_exploration_count += 1
                else:
                    rand_la.current_action = 0
                    rand_la.bad_candidate_exploration_count += 1


        rand_la.inc_amount = budget/ 50#self.la_m_depth

    def get_explored_element(self, choices_range, i):
        """select the element to be explored and evaluated inside the sample

        Parameters
        -------------------
        choices_range: range
                the range of all other LAs (users) indices that were not considered the top LAs with higher probs
        i: int
                the iteration number which is considered the time step that will be used in the exponentional decay function
                for exploration
        Returns
        -------------------
        int
            an index to be selected from the choices range """
        if 1 * (1-self.explore_decay_factor)**i > 0.0001:
            rand_index = random.choice(choices_range)
            return rand_index
        print('######################')
        print('EXPLORATION DECAYED')
        print('######################')
        return -1

    def is_odd(self, i):
        """helper function to check if the current iteration number index is an even or odd

        Parameters
        -----------------
        i: int
            current iteration number, used to check if the number is odd or even so sample size is extended by one if even
             for evaluating and exploring new element(s)"""
        if i % 2 == 0:
            return False # Even
        else:
            return True # Odd

    def simulate_team(self, i, original_baselines_true_news, fake_news_data,
                      fake_news_timestamps, users_to_calc_results, adjacency, adjacency_explicit, decay, stage, random_sampling,
                      sample_size):
        """function to simulate a true news campaign hawkes process with a modified base intensity for a sampled LA team in
        the network at a stage/ iteration

        Parameters
        -------------------
        i: int
                    current number for the current iteration so we know if it is an even number, we should reform a new
                     random sample
         original_baselines_true_news: list
                    list of original base intensities before mitigation and adding values to them
         fake_news_data: list
                    a list of n stages fake news exposures averages in each stage
         fake_news_timestamps: list
                    list of numpy arrays as each user generated timestamps of fake news
         users_to_calc_results: list
                    a list of n stages lists where each stage list has user ids that are important to calculate
                    their mitigation results as they are the highest exposed to fake news at that stage
         adjacency: numpy array
                    a matrix of n_users X n_users and will be used as parameter for the mitigation simulations
         adjacency_explicit: numpy array
                    a matrix with the network explicit given influencing relationship, it is used to calculate the impacts of
                    both true and fake news after mitigation to evaluate how good was the intervention (incentivization)
         decay: float
                    another parameter required for the mitigation simulation, parameters should be the same as those
                    were given in the simulations before mitigation for consistent results
        stage: int
                the id of the stage to compare results before and after mitigation
        random_sampling: Boolean
                a boolean flag to indicate if we should random a small sized sample for evaluating the objective function
                and running the simulation or not. It is usually going to be true for light weight computation but at certain
                iterations it will be a full network simulation to evaluate the current learned LAs (users) that supposed to be
                the most fitting ones for a mitigation strategy.
        Returns
        ------------------
        float
            a float representing the mitigated fake news campagin measure, the less the value is, the better the sample is
        """
        if type(fake_news_timestamps) is dict:
            fake_news_timestamps = list(fake_news_timestamps.values())
        if random_sampling:
            sub_adjacency, sub_base_lines, sub_users_on_focus, sub_adjacency_explicit, sub_fake_news_stamps, mapping = \
                self.transformto_subnetwork(adjacency, original_baselines_true_news, users_to_calc_results, fake_news_timestamps,
                                            adjacency_explicit, sample_size)

            hk = SimuHawkesExpKernels(sub_adjacency, decay, self.change_user_intensity(sub_base_lines, mapping),
                                      self.end_time, seed=0, verbose=False)
            hk.threshold_negative_intensity(allow=True)
            hk.simulate()
            mitigated_simulation_timestamps = hk.timestamps
            results_after_mitigation = get_high_exposures_users(sub_fake_news_stamps, mitigated_simulation_timestamps,
                                                                sub_adjacency_explicit, self.realizations_n,
                                                                self.realizations_bounds, sub_users_on_focus)
            true_news_data_after_mitigation = get_stages_avg_exposures(results_after_mitigation)[0]
            corr = calc_correlation(fake_news_data, true_news_data_after_mitigation)
            diff = calc_difference(fake_news_data, true_news_data_after_mitigation)
            result1 = corr[stage]
            result2 = diff[stage]

        else:
            hk = SimuHawkesExpKernels(adjacency, decay, self.change_user_intensity(original_baselines_true_news, None),
                                      self.end_time, seed=0, verbose=False)
            hk.threshold_negative_intensity(allow=True)
            hk.simulate()
            mitigated_simulation_timestamps = hk.timestamps
            results_after_mitigation = get_high_exposures_users(fake_news_timestamps, mitigated_simulation_timestamps,
                                                                adjacency_explicit, self.realizations_n,
                                                                self.realizations_bounds, None)
            true_news_data_after_mitigation = get_stages_avg_exposures(results_after_mitigation)[0]
            corr = calc_correlation(fake_news_data, true_news_data_after_mitigation)
            diff = calc_difference(fake_news_data, true_news_data_after_mitigation)
            result1 = corr[stage]
            result2 = diff[stage]
            #print('la', diff)
        return result2, result1

    def transformto_subnetwork(self, network_adjacency, original_baselines_whole_network, users_on_focus, whole_fake_stamps,
                               network_explicit_adjacency, sample_size):
        """this function is responsible for the sub network to be chosen as a sample for the evaluation of the objective
        function as evaluating over the whole network is computationally expensive

        Parameters
        -------------------
        network_adjacency: numpy array
                        the matrix of all network adjacency matrix estimated from Hawkes estimator algorithms and to be
                        shortened to a smaller sample size
        original_baselines_whole_network: list
                        list of original estimated base intensities of true news for the whole network users, and this to be shortened
                        to a smaller sample size
        users_on_focus: list of lists
                        a list of n stages lists of user ids in each stage to calculate the objective function for,
                        this to be shortened according to the smaller sample size included user ids
        whole_fake_stamps: list
                        list of users fake news timestamps so that we extract only the fake news timestamps for the selected users
        network_explicit_adjacency: numpy array
                        matrix of the explicit influence info from the data, we select and transform it to only the filtered
                        users for the sized simulation size, we do the exact as we do with the network implicit adjacency param

        Returns
        ---------------------
        numpy array
                the filtered smaller network adjacency
        list
                the filtered smaller baselines
        list
                the returned filtered list of n stages lists of filtered users
        list
                the filtered fake news time stamps
        numpy array
                the filtered smaller explicit network adjacency
        """
        adjacency_sample = np.zeros((sample_size, sample_size), dtype=float)
        adjacency_sample_explicit = np.zeros((sample_size, sample_size), dtype=float)
        sample_baselines = np.zeros(sample_size, dtype=float)
        randomly_selected_users_indices = []
        for i in range(self.team_size):
            randomly_selected_users_indices.append(self.current_to_exploit_las_ids[i])

        while len(randomly_selected_users_indices) < sample_size:
            ind = random.choice(range(0, len(self.network_LAs)))
            if ind not in self.current_to_exploit_las_ids and ind not in randomly_selected_users_indices:
                randomly_selected_users_indices.append(ind)

        mapping = []
        indexer_1 = 0
        for index1 in randomly_selected_users_indices:
            indexer_2 = 0
            for index2 in randomly_selected_users_indices:
                adjacency_sample[indexer_1][indexer_2] = self.get_matrix_sample_value(index1, index2, network_adjacency)
                indexer_2 += 1
            mapping.append((indexer_1, index1))
            indexer_1 += 1

        indexer_1 = 0
        for index1 in randomly_selected_users_indices:
            indexer_2 = 0
            for index2 in randomly_selected_users_indices:
                adjacency_sample_explicit[indexer_1][indexer_2] = self.get_matrix_sample_value(index1, index2, network_explicit_adjacency)
                indexer_2 += 1
            indexer_1 += 1


        for user_id, base_intensity in enumerate(original_baselines_whole_network):
            if user_id in randomly_selected_users_indices:
                sample_baselines[self.fetch_mapped_id(user_id, mapping)] = base_intensity + self.sensetivity_eval_param

        sample_users_on_focus = []
        for stage in users_on_focus:
            stage_ids = []
            for u_id in stage:
                if u_id in randomly_selected_users_indices:
                    stage_ids.append(self.fetch_mapped_id(u_id, mapping))
            sample_users_on_focus.append(stage_ids)

        sampled_fake_news_stamps = []
        for x in range(sample_size):
            sampled_fake_news_stamps.append(None)

        for e, np_arr in enumerate(whole_fake_stamps):
            for m in mapping:
                if e == m[1]:
                    sampled_fake_news_stamps[m[0]] = np_arr

        return adjacency_sample, sample_baselines, sample_users_on_focus, adjacency_sample_explicit, sampled_fake_news_stamps, mapping

    def fetch_mapped_id(self, old_id, mapping_info):
        """

        :param old_id: int
                        for the original index in the whole network for a certain user
        :param mapping_info: list of tuples
                        the (new index, old index) info
        :return: int
                        the new index to fit in the new sample size
        """
        for info in mapping_info:
            if info[1] == old_id:
                return info[0]

    def fetch_mapped_oldid(self, old_id, mapping_info):
        """

        :param old_id: int
                        for the original index in the whole network for a certain user
        :param mapping_info: list of tuples
                        the (new index, old index) info
        :return: int
                        the new index to fit in the new sample size
        """
        for info in mapping_info:
            if info[0] == old_id:
                return info[1]

    def get_matrix_sample_value(self, index_1, index_2, whole_network_adjacency):
        """
        helper function
        :param index_1: int
        :param index_2: int
        :param whole_network_adjacency: numpy array
        :return: float value for the adjacency matrix
        """
        return whole_network_adjacency[index_1][index_2]

    def change_user_intensity(self, original_baselines_before_mitigation, mapping=None):
        """function to loop through users baselines and selects the given user id to modify its base intensity

        Parameters
        -----------------
        original_baselines_before_mitigation: list
                                            list of all network users baselines, only current LA team members (users9
                                            will be modified
        mapping: list
                                            if not none, indicates whether we should do the modification according to a new randomized
                                            ids which require the mapping of the subnetwork transformation, the list by then
                                            is a list of tuples  (new_id, old_id)

        Returns
        -------------------
        list
            a list of n users baselines (base intensities) after modifying some of them
        """
        ids = []
        for id, _ in self.team_members_values:
            if mapping == None:
                ids.append(id)
            else:
                ids.append(self.fetch_mapped_id(id, mapping))

        baselines = np.zeros(len(original_baselines_before_mitigation), dtype=float)
        for e, value in enumerate(original_baselines_before_mitigation):
            if e in ids:
                if mapping == None:
                    baselines[e] = value + self.get_LA_obj(e).current_state
                else:
                    baselines[e] = value + self.get_LA_obj(self.fetch_mapped_oldid(e, mapping)).current_state
            else:
                baselines[e] = value
        #print(ids)
        #print('from LA', baselines)
        return baselines

    def get_LA_obj(self, la_id):
        """helper function to get the LA object given the integer value if its id

        Parameters
        ------------------
        la_id: int
                the object id

        Returns
        ------------------
        LA object
                the desired LA object
        """
        for la in self.network_LAs:
            if la.id == la_id:
                return la

    def update_LA_states(self, feedback, current_budget, all_budget):
        """this function to update all underlying LAs in the current LA team sample, either to accept the action and
        reward the LA by moving towards more incentivization states or deny the action and stay at the current state, that
        applies also on updating the probabilities of the LA action either positively or negatively"""

        for la_id, _ in self.team_members_values:
            if feedback == 1:
                self.get_LA_obj(la_id).increase_state(current_budget, all_budget)

            elif feedback == -1:
                self.get_LA_obj(la_id).decrease_state()
            else:
                return

        # for la_id, _ in self.team_members_values:
        #     if feedback > .5:
        #         if la_id in self.current_explored_LAs:
        #             self.get_LA_obj(la_id).accept_act()
        #             if len(self.current_replaced_LAs) > 0:
        #                 self.get_LA_obj(self.current_replaced_LAs[0][0]).deny_act()
        #     else:
        #         if la_id in self.current_explored_LAs:
        #             self.get_LA_obj(la_id).deny_act()
        #             if len(self.current_replaced_LAs) > 0:
        #                 self.get_LA_obj(self.current_replaced_LAs[0][0]).accept_act()


class Network(object):
    """this class is to define a network of competitive LA teams, each team is a set of LAs, each LA team represents
    a random selected sample of las (mimic users), and each sample is evaluated till we found the best sample that
    minimize an objective function which represents the performance of the mitigation as less values mean less effect
    of fake news."""
    def __init__(self, lambdaR, budget, network_size, team_size, epochs, original_baselines_true_news,
                  decay_mis, decay_norm, adjacency_explicit, adjacency_weighted, realizations_n, realizations_bounds,
                 la_m_depth, fake_news_data, fake_news_stamps,
                 users_to_calc_results, sample_size, sens_param):
        """the initiator method for the class, it is used to define the structure of the network and each LA team LA memory depth
        and state limit according to the given budget as well, it also defines the current actual(from simulation) impacts
         between true and fake news campaigns so it can use this information to evaluate the LA team performance

         Parameters
         -------------------
         lambdaR: float
                    the individual LA action selection probability update factor
         memory_depth: int
                    number of steps the indivifual LA in the LA team in the network can go far, that amount should be
                    equal for all LAs in all teams accros the network, and should be propotional to the allowed buget
                     and the number of LAs on all teams
         network_size: int
                    number of las in the network from where an LA team can be sampled
         team_size: int
                    the length of a list of candidate LAs (users) inside each la team and by which will be examined at each
                     iteration
         n_iterations : int
                    number iterations the sampling will be formed at each one, each iteration the sample (la team)
                    is changed, hopefully till convergence to the winner la team before the number of iterations ends.
         original_baselines_true_news: list
                    list of original base intensities before mitigation and adding values to them
         fake_news_data: list
                    a list of n stages fake news exposures averages in each stage
         true_news_data_before_mitigation: list
                    a list of n stages true news exposures averages in each stage
         fake_news_timestamps: list
                    list of numpy arrays as each user generated timestamps of fake news
         users_to_calc_results: list
                    a list of n stages lists where each stage list has user ids that are important to calculate
                    their mitigation results as they are the highest exposed to fake news at that stage
         realizations: int
                    number of stages that the network should learn at each the best subset of users to use for
                    mitigation
         adjacency: numpy array
                    a matrix of n_users X n_users and will be used as parameter for the mitigation simulations
         adjacency_explicit: numpy array
                    a matrix with the network explicit given influencing relationship, it is used to calculate the impacts of
                    both true and fake news after mitigation to evaluate how good was the intervention (incentivization)
         decay: float
                    another parameter required for the mitigation simulation, parameters should be the same as those
                    were given in the simulations before mitigation for consistent results
          """
        self.network_size = network_size
        self.network_lambdaR = lambdaR
        self.budget = budget
        self.LA_teams_results = []
        self.epochs = epochs
        self.team_size = team_size
        self.original_baselines_true_news = original_baselines_true_news
        self.realizations = realizations_n
        self.realizations_bounds = realizations_bounds
        self.adjacency = adjacency_explicit
        self.adjacency_weighted = adjacency_weighted
        self.decay = decay_mis
        self.LAs = []
        self.end_time = 28000
        self.probs_vector = []
        self.probs_vectors = []
        self.states_values = []
        self.mu = None
        self.la_m_depth = la_m_depth
        self.fake_news_data = fake_news_data
        self.fake_news_timestamps = fake_news_stamps
        self.users_to_calc_results = users_to_calc_results
        self.sample_size = sample_size
        self.sensetivity_eval_param = sens_param

    def reset(self, scheme):
        """method to build and initiate the network, and assign the lambdaR and memory depth to each LA before sampling
        an LA team"""
        self.LAs = []
        self.probs_vectors = []
        if scheme == 0:
            for id, node in enumerate(range(self.network_size)):
                la = LA_RAND_WALK(self.network_lambdaR, self.budget, id)
                self.LAs.append(la)
    def converged(self):
        """this function checks the probs vector to see if it converged"""
        if len(self.probs_vectors) > 2:
            if self.probs_vectors[-3] == self.probs_vectors[-2] == self.probs_vectors[-1]:
                return True
        return False

    def la_random_walk_learner(self, stage):
        stage_expected_values = []
        self.reset(0)
        self.states_values = []
        xs = []
        ys_rand_walk = []
        for i in range(self.epochs):
            for iter in range(len(self.LAs)):
                la_team = LATeam(self.LAs, self.team_size, self.realizations, self.realizations_bounds,
                                 self.end_time, self.la_m_depth, self.sensetivity_eval_param)
                la_team.team_size = 1
                la_team.sampleLATeam(0, iter, self.budget)
                la_ids_set = la_team.team_members_values
                random_sampling = True
                if iter == len(self.LAs) - 1:
                    test_probs = []
                    for l in self.LAs:
                        test_probs.append((l.id, l.good_candidate_prob))
                    la_team.team_members_values = sorted(test_probs, key=lambda tup: tup[1], reverse=True)
                    la_team.team_size = 5
                    la_team.current_to_exploit_las_ids = []
                    la_ids_set = la_team.team_members_values
                    for lear_id, lear_in_prob in la_team.team_members_values:
                        la_team.current_to_exploit_las_ids.append(lear_id)
                    random_sampling = False

                obj_func_value = la_team.simulate_team(i, self.original_baselines_true_news, self.fake_news_data,
                                                       self.fake_news_timestamps, self.users_to_calc_results,
                                                       self.adjacency_weighted, self.adjacency, self.decay, stage, random_sampling,
                                                       self.sample_size)
                if random_sampling == True:
                    expected_v = obj_func_value[0] * la_ids_set[0][1]
                    stage_expected_values.append(round(expected_v, 1))
                    la_team.get_LA_obj(la_ids_set[0][0]).expected_values.append(expected_v)
                    if len(stage_expected_values) > 2:
                        #print(stage_expected_values[-1], stage_expected_values[-2], obj_func_value[0], la_ids_set[0][1])
                        if stage_expected_values[-1] < stage_expected_values[-2]:
                            # print('REWARD', la_ids_set[0][0])
                            # time.sleep(1)
                            la_team.get_LA_obj(la_ids_set[0][0]).update_la_probs(1)
                        else:
                            # print('PENALTY', la_ids_set[0][0])
                            # time.sleep(1)
                            la_team.get_LA_obj(la_ids_set[0][0]).update_la_probs(0)

                        feedback = self.get_feedback(la_team.get_LA_obj(la_ids_set[0][0]))
                        la_team.update_LA_states(feedback, self.calc_current_budget(), self.budget)
                        self.update_probs_vector()
                network_set = []
                for learau in self.LAs:
                    network_set.append((learau.id, round(learau.good_candidate_prob, 4)))


            #self.print_status(i, obj_func_value[0], la_ids_set, network_set)
            xs.append(i)
            ys_rand_walk.append(obj_func_value[0])
            self.LA_teams_results.append(obj_func_value[0])
            count = 0

        return xs, ys_rand_walk, la_team

    def calc_mu(self, list_n, list_nplus1, list_star):
        newlist2= list(zip(list_n, list_star))
        newlist1 = list(zip(list_nplus1, list_star))
        vector_1 = []
        vector_2 = []
        for elem1, elem2 in newlist2:
            vector_2.append(elem1 - elem2)
        for ele1, ele2 in newlist1:
            vector_1.append(ele1 - ele2)

        vector_1_norm = 0
        for v1 in vector_1:
            vector_1_norm += v1**2
        vector_1_norm = math.sqrt(vector_1_norm)

        vector_2_norm = 0
        for v2 in vector_2:
            vector_2_norm += v2**2
        vector_2_norm = math.sqrt(vector_2_norm)
        if vector_2_norm != 0:
            return vector_1_norm/ vector_2_norm
        else:
            return None

    def check_convergence_vector(self):
        if len(self.states_values) > 4:
            if self.states_values[-5] == self.states_values[-4] and self.states_values[-5] == self.states_values[-3] \
                    and self.states_values[-5] == self.states_values[-2]  and self.states_values[-5] == self.states_values[-1]:
                return self.states_values[-5]
            else:
                return self.states_values[-1]

    def run(self, stage):
        """method to run a knapsack optimization on the network sampled LA teams per iteration, it loops through each stage as an independent
        knapsack problem so at each stage, it loops through n iterations trying to converge to a certain subset of users
        when used to modify their base intensities, the mitigation results in the minimum possible value (objective function)"""
        d_m_v_rw = {}

        print('Starting Optimization over stage ', str(stage))
        xs, ys_rand_walk, la_team_rand_walk = self.la_random_walk_learner(stage)
        test_probs = []
        for l1 in self.LAs:
            test_probs.append((l1.id, l1.good_candidate_prob))
        sorted_las = sorted(test_probs, key=lambda tup: tup[1], reverse=True)
        print('##############################')
        for s in sorted_las:
            d_m_v_rw[s[0]] = la_team_rand_walk.get_LA_obj(s[0]).current_state

        return d_m_v_rw

    def calc_current_budget(self):
        budget = 0
        for la in self.LAs:
            budget += la.current_state
        return budget


    def get_feedback(self, la_obj):
        """function to evaluate the potential of a current LA team sample, so if its performance was better than its ancestor
        , the feedback should be positive for that sample, otherwise, the previous sample is preferred and its probs are
         positively updated"""
        if la_obj.good_candidate_prob > la_obj.bad_candidate_prob  and self.calc_current_budget() + la_obj.inc_amount <= self.budget:
            #print(la_obj.id, 'INCREASING STATE', la_obj.good_candidate_prob,la_obj.good_candidate_prob > la_obj.bad_candidate_prob,  self.calc_current_budget() + la_obj.inc_amount)
            #time.sleep(1)
            return 1
        elif la_obj.good_candidate_prob < la_obj.bad_candidate_prob:
            #print(la_obj.id, 'DECREASING STATE', la_obj.good_candidate_prob, la_obj.good_candidate_prob > la_obj.bad_candidate_prob,  self.calc_current_budget() + la_obj.inc_amount)
            #time.sleep(1)
            return -1
        else:
            #print(la_obj.id, 'Neutral to STATE', la_obj.good_candidate_prob, la_obj.good_candidate_prob > la_obj.bad_candidate_prob,  self.calc_current_budget() + la_obj.inc_amount)
            #time.sleep(1)
            return 0

    def print_status(self, i, result, la_ids_set, network_set):
        """a method to print the current network status and the current adaptive learned sample with the current
        iteration number and the value of the objective function

        Parameters
        -----------------------
        i: int
            the current iteration number
        result: float
            the current value of the objective function to be minimized
        la_ids_set: list
            list of tuples (user id, incentivization probability), each represents the current learned sample
        network_set: list
            list of tuples (user id, incentivization probability), each represents the current user status in the whole network
        """

        for _ in range(len(network_set)-len(la_ids_set)):
            la_ids_set.append(('', ''))

        sample = np.array(la_ids_set)
        network = np.array(network_set)
        status = [" ".join(item) for item in np.column_stack((network, sample)).astype(str)]
        print(*status, sep='\n')
        print(i, round(result, 4))

    def update_probs_vector(self):
        """this function updates the incentivization action propablities for all LAs in the network so that it can be checked
        for convergence after each iteration updates."""

        self.probs_vector = []
        for la in self.LAs:
            self.probs_vector.append(la.good_candidate_prob)
        self.probs_vectors.append(self.probs_vector)

