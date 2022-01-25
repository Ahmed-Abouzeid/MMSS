import random
from knapsack import Knapsack
from social_network import SocialNetwork

class Random_Cont(object):

    def __init__(self, b, MU, responsive_users = None):
        """the constructor defines the base intensity to be controlled and the amount of user incintivization budget"""

        self.b = b
        self.MU = MU
        self.resposnive_users = responsive_users

    def control(self):
        """loop till the budget is consumed, the consumption is random per user"""

        controlled_MU = self.MU
        user_indices = [i for i in range(len(self.MU))]
        budget_counter = 0
        if self.resposnive_users:
            user_indices = self.resposnive_users

        while True:
            for id, _ in enumerate(self.MU):
                if id in user_indices:
                    incent_amount = random.uniform(0, self.b)
                    budget_counter += incent_amount
                    if budget_counter <= self.b:
                        controlled_MU[id] += incent_amount
                    else:
                        return controlled_MU


class Uniform_Cont(object):

    def __init__(self, b, MU, responsive_users=None):
        """the constructor defines the base intensity to be controlled and the amount of user incintivization budget"""

        self.b = b
        self.MU = MU
        self.resposnive_users = responsive_users

    def control(self):
        """loop till the budget is consumed, the consumption is uniformly assigned per user"""

        controlled_MU = self.MU.copy()
        if self.resposnive_users:
            incent_amount = self.b/len(self.resposnive_users)
            for id in self.resposnive_users:
                controlled_MU[id] += incent_amount
        else:
            incent_amount = self.b / len(self.MU)
            for i in range(len(self.MU)):
                controlled_MU[i] += incent_amount

        return controlled_MU


class Weak_Cont(object):

    def __init__(self, b, MU, influence_relations, responsive_users= None):
        """the constructor defines the base intensity to be controlled and the amount of user incintivization budget"""

        self.b = b
        self.MU = MU
        self.influence_relations = influence_relations
        self.responsive_users = responsive_users

    def control(self):
        """loop till the budget is consumed, the consumption is assigned to top explicit influence
         users in the network. Influence info is extracted from top users that their posted content was
         shared more by others in the past before the moment of control. We call the method weak because these top
         influencers can be main spreader of misinformation and no learning was applied to learn their authenticity"""

        controlled_MU = self.MU.copy()
        if self.responsive_users:
            influencers_weights = []
            for influencer_id in self.influence_relations.keys():
                if influencer_id in self.responsive_users:
                    influence_weight = len(self.influence_relations[influencer_id])
                    influencers_weights.append((influencer_id, influence_weight))
        else:
            influencers_weights = []
            for influencer_id in self.influence_relations.keys():
                influence_weight = len(self.influence_relations[influencer_id])
                influencers_weights.append((influencer_id, influence_weight))

        if not self.responsive_users:
            for id, weight in influencers_weights:
                # to assign incentivization to each user based on how much the user is an influencer
                controlled_MU[id] += self.b/len(influencers_weights) #* weight

        else:
            for id, weight in influencers_weights:
                if id in self.responsive_users:
                    # to assign incentivization to each user based on how much the user is an influencer
                    controlled_MU[id] += self.b/len(influencers_weights) #* weight

        return controlled_MU


class AI_Cont(object):

    def __init__(self, pred_mis_timestamps, pred_norm_timestamps, adjaceny_matrix,
                 realization_id, config, norm_MU, norm_A, normal_timestamps_before_simu,
                 responsive_users = None):
        self.config = config
        self.MU = norm_MU
        self.A = norm_A
        self.pred_mis_stamps = pred_mis_timestamps
        self.pred_norm_stamps = pred_norm_timestamps
        self.adjacency_matrix = adjaceny_matrix
        self.r_id = realization_id
        self.bounds = config.realizations_bounds
        self.responsive_users = responsive_users
        self.parallel = config.parallel
        self.normal_timestamps_before_simu = normal_timestamps_before_simu
        self.verbose = config.verbose

    def control(self):
        """controlling the network with learned hidden influence users. Modeling a knapsack problem"""

        sn = SocialNetwork(self.pred_mis_stamps, self.pred_norm_stamps, self.adjacency_matrix, self.r_id,
                            self.bounds, self.config, self.MU, self.A, self.normal_timestamps_before_simu)
        ks = Knapsack(range(len(self.MU)), self.config, sn)
        las, consumed_budget = ks.random_walk(self.verbose, self.parallel)
        del sn, ks

        controlled_MU = self.MU.copy()
        for user_id, _ in enumerate(self.MU):
            controlled_MU[user_id] += las[user_id].current_state

        return controlled_MU, consumed_budget
