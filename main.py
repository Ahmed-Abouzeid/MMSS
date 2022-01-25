from argparse import ArgumentParser
from utils import get_network_metadata, get_categorized_timestamps, shrink_network, create_synthetic_network
from demo import run_experiments, run_avg_method
import time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retweet_path', type=str, default="data/raw/covid19/tree")
    parser.add_argument('--labels_path', type=str, default="data/raw/covid19/label.txt")
    parser.add_argument('--realizations_n', type=int, default=5)
    parser.add_argument('--realizations_bounds', type=object, default = [x * 7200 for x in [1, 2, 3, 4, 5]])
    parser.add_argument('--decay_factor_mis', type=float, default=.8)
    parser.add_argument('--decay_factor_norm', type=float, default=1)
    parser.add_argument('--pred_realizations_period', type=int, default=7200)  # we predict next 2 hours
    parser.add_argument('--pred_start_time', type=int, default=28800)
    parser.add_argument('--shrinked_network_size', type=int, default=100)
    parser.add_argument('--real_network_end_time', type=int, default=28800)
    parser.add_argument('--average_out_runs', type=int, default=1)
    parser.add_argument('--budget', type=float, default=.18)
    parser.add_argument('--step', type=float, default=0.000001)
    parser.add_argument('--balance_factor', type=float, default=1.3)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--la_m_depth', type=int, default=900)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--synthetic', type=bool, default=False)
    parser.add_argument('--synthetic_skewness', type=float, default=.07)
    parser.add_argument('--verbose', type=bool, default = True)
    parser.add_argument('--parallel', type=bool, default = False)
    parser.add_argument('--is_avg_method', type=bool, default = False)
    parser.add_argument('--show_simu_error', type=bool, default = True)
    parser.add_argument('--graphs', type=bool, default = False)
    parser.add_argument('--sampling', type=bool, default = True)
    parser.add_argument('--sample_size', type=int, default = 200)
    parser.add_argument('--adjacents_sample_size', type=int, default = None)
    config = parser.parse_args()
    if not config.synthetic:
        all_users, _, _,  tweeting_time, influence_relations = get_network_metadata(config.retweet_path)
        mis_timestamps, normal_timestamps = get_categorized_timestamps(tweeting_time, config.labels_path)

        # we resize the network to avoid slow computation issues if necessary

        mis_timestamps, normal_timestamps, tweeting_time, influence_relations = \
            shrink_network(mis_timestamps, normal_timestamps, tweeting_time, influence_relations,
                           config.shrinked_network_size)

    else:
        mis_timestamps, normal_timestamps, tweeting_time, influence_relations =create_synthetic_network(config.synthetic_skewness,
                                                                    config.shrinked_network_size,
                                                                config.pred_start_time-100, config.labels_path)

    run_experiments(config, mis_timestamps, normal_timestamps, tweeting_time, influence_relations, config.is_avg_method)







