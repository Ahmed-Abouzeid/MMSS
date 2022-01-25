from tick.hawkes import HawkesADM4, SimuHawkesExpKernels
from utils import transform_to_sub_network, sample_network, fetch_mapped_id
import numpy as np


class Hawkes(object):

    def __init__(self, timestamps, decay_factor, config):
        self.timestamps = timestamps
        self.decay_factor = decay_factor
        self.pred_realizations_period = config.pred_realizations_period
        self.pred_start_time = config.pred_start_time
        self.real_network_end_time = config.real_network_end_time
        self.config = config

    def estimate_params(self, n):
        """estimate a parametric Hawkes process. The parameters to be estimated are the base intensity and
        the hidden influence matrix"""

        # n indicates number of runs before we average out the error from the estimations
        timestamps = []
        for user_stamps in self.timestamps:
            u_stamps = []
            for t in user_stamps:
                if t <= self.real_network_end_time:
                    u_stamps.append(t)
            timestamps.append(np.array(u_stamps))

        all_A = []
        all_MU = []
        for i in range(n):
            print('Hawkes parameters estimating. #Run:', i)
            learner = HawkesADM4(self.decay_factor, n_threads=0)
            learner.fit(timestamps, adjacency_start=None)
            all_A.append(learner.adjacency)
            all_MU.append(learner.baseline)

        MU = np.zeros(len(all_MU[0]))
        A = np.zeros((len(MU), len(MU)))
        for mu in all_MU:
            MU += mu
        for a in all_A:
            A += a

        return MU/n, A/n

    def simulate(self, MU, A, norm_time_stamps_before_control, verbose=False, sampling = False, controlled_user_id = None):
        """after learning the process parameters, this method simulates the dynamics of social network using Hawkes
        exponential decay intensity function simulator object"""

        # A is the estimated hidden influence matrix
        # MU is the estimated base intensity
        if sampling:
            sample_users_ids = sample_network(controlled_user_id, len(MU), self.config.sample_size)
            A, MU, mapping_info = transform_to_sub_network(A, MU, sample_users_ids)

        mhp = SimuHawkesExpKernels(A, self.decay_factor, MU, self.pred_realizations_period, seed=0, verbose=False)
        mhp.threshold_negative_intensity(allow=True)
        mhp.simulate()

        pred_timestamps = {}
        for e, user_stamps in enumerate(mhp.timestamps):
            x = user_stamps + self.pred_start_time
            pred_timestamps.update({e: x})
        if sampling:
            for k in pred_timestamps.keys():
                for sample_id, original_id in mapping_info:
                    if k == sample_id:
                        norm_time_stamps_before_control[original_id] = pred_timestamps[sample_id]
                        if norm_time_stamps_before_control[original_id].all() != pred_timestamps[sample_id].all():
                            print('updated ', original_id, sample_id)
                            print(norm_time_stamps_before_control)
                            print(pred_timestamps[sample_id])
                            exit()
                    break
            return norm_time_stamps_before_control
        else:
            return pred_timestamps
