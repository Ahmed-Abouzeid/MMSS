from model_predictor import Hawkes
from utils import create_simu_vis_graph, format_hawkes_stamps, add_predicted_tweets, merge_timestamps,\
    calc_network_fairness, calc_avg_abs_err
from model_control import Uniform_Cont, AI_Cont
import numpy as np
from Comparable_Intervention import *
from Comparable_Network_Measures import *
import time


def run_experiments(config, mis_timestamps, normal_timestamps, tweeting_time, influence_relations, is_avg_method):
    no_keys_mis_timestamps = format_hawkes_stamps(mis_timestamps)
    no_keys_normal_timestamps = format_hawkes_stamps(normal_timestamps)
    """runs all experiments: LA, uniform, and before mitigation"""

    hawkes_mis = Hawkes(no_keys_mis_timestamps, config.decay_factor_mis, config)
    mis_MU, mis_A = hawkes_mis.estimate_params(config.average_out_runs)
    pred_mis_timestamps = hawkes_mis.simulate(mis_MU, mis_A, None, verbose=False, sampling=False)
    if config.show_simu_error:
        err = calc_avg_abs_err(pred_mis_timestamps, mis_timestamps, realization_bounds=config.realizations_bounds)
        print('Mis Content Simulation Error: ', err)
    del hawkes_mis

    hawkes_normal = Hawkes(no_keys_normal_timestamps, config.decay_factor_norm, config)
    norm_MU, norm_A = hawkes_normal.estimate_params(config.average_out_runs)
    pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
    if config.show_simu_error:
        err = calc_avg_abs_err(pred_norm_timestamps, normal_timestamps, realization_bounds=config.realizations_bounds)
        print('Norm Content Simulation Error: ', err)
    merged_mis_timestamps, merged_norm_timestamps = merge_timestamps(mis_timestamps, pred_mis_timestamps,
                                                                     normal_timestamps, pred_norm_timestamps)
    adjaceny_matrix = np.zeros((len(norm_A), len(norm_A)))
    old = norm_MU.copy()
    # iterate through rows
    for i in range(len(norm_A)):
        # iterate through columns
        adjaceny_matrix[i][i] += 1
        for j in range(len(norm_A[0])):
            if i != j:
                if norm_A[i][j] + mis_A[i][j] > 0:
                    adjaceny_matrix[i][j] += 1

    all_percs_la, all_percs_uniform, all_percs_before = [], [], []
    consumed_budgets = []
    all_fairness_la, all_fairness_uniform, all_fairness_before = [], [], []
    for i in range(3):
        if i == 0:
            all_percs_la = []
            all_fairness_la = []
            all_compu_speeds = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                t_1 = time.time()
                if is_avg_method:
                    norm_MU, la_consumed_budget = run_avg_method(config, merged_norm_timestamps,
                                                                 merged_mis_timestamps, adjaceny_matrix, norm_MU,
                                                                 norm_A)
                else:
                    ai_cont = AI_Cont(merged_mis_timestamps, merged_norm_timestamps, adjaceny_matrix, 4, config,
                                      old, norm_A, normal_timestamps)
                    norm_MU, la_consumed_budget = ai_cont.control()
                t_2 = time.time()
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps_after_control, merged_norm_timestamps_after_control = merge_timestamps(
                    mis_timestamps, pred_mis_timestamps,
                    normal_timestamps,
                    pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps_after_control,
                                                              merged_mis_timestamps_after_control,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)

                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps_after_control,
                                                            merged_norm_timestamps_after_control,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_la.append(mis_perc_last_stage)
                all_fairness_la.append(network_fairness_loss)
                all_compu_speeds.append(t_2-t_1)
                consumed_budgets.append(la_consumed_budget)

        elif i == 1:
            all_percs_uniform = []
            all_fairness_uniform = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                uniform_cont = Uniform_Cont(np.mean(consumed_budgets), old)
                norm_MU = uniform_cont.control()
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps_after_control, merged_norm_timestamps_after_control = merge_timestamps(
                    mis_timestamps, pred_mis_timestamps,
                    normal_timestamps,
                    pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps_after_control,
                                                              merged_mis_timestamps_after_control,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)
                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps_after_control,
                                                            merged_norm_timestamps_after_control,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_uniform.append(mis_perc_last_stage)
                all_fairness_uniform.append(network_fairness_loss)

        elif i == 2:
            all_percs_before = []
            all_fairness_before = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                norm_MU = old
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps, merged_norm_timestamps = merge_timestamps(mis_timestamps, pred_mis_timestamps,
                                                                                 normal_timestamps,
                                                                                 pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps, merged_mis_timestamps,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)

                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps, merged_norm_timestamps,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_before.append(mis_perc_last_stage)
                all_fairness_before.append(network_fairness_loss)

    print('\n########################################################################################################')
    print('Averaged Results of Mis Perc Last Stage (LA - Uniform - Before ANy Intervention):',
          np.mean(all_percs_la), np.mean(all_percs_uniform), np.mean(all_percs_before))
    print('Averaged Results of achieved network fairness (LA - Uniform - Before ANy Intervention):',
          np.mean(all_fairness_la), np.mean(all_fairness_uniform), np.mean(all_fairness_before))
    print('Consumed Budget On Average: ', np.mean(consumed_budgets))
    print('Computation Speed On Average (Seconds): ', np.mean(all_compu_speeds))
    print('Computation Speed On Average (Minutes): ', np.mean(all_compu_speeds)/60)
    print('STD for Fairness Error: ', np.std(all_fairness_la))
    print('STD for Final Mis Info Perc: ', np.std(all_percs_la))
    print('STD for Mis Info Perc Before Intervention: ', np.std(all_percs_before))
    print('STD for Compu Speed: ', np.std(all_compu_speeds))
    print('########################################################################################################')


def run_avg_method(config, merged_norm_timestamps, merged_mis_timestamps, adjaceny_matrix, norm_MU, norm_A):
    '''runs other mitigation method in the literature (Abouzeid et.al 2021) where no fairness is considered'''
    """runs all experiments: LA, uniform, and before mitigation"""

    results_before_mitigation = get_high_exposures_users(merged_norm_timestamps, merged_mis_timestamps, adjaceny_matrix,
                                                         5, config.realizations_bounds, None)
    mitigated_users_per_stage = get_selected_users_per_stage(results_before_mitigation)

    fake_news_data = get_stages_avg_exposures(results_before_mitigation)[1]

    sample_size = 200
    sens_param = 0.001
    epochs = 20
    n = Network(.98, config.budget, config.shrinked_network_size, 1, epochs, norm_MU,
                config.decay_factor_mis, config.decay_factor_norm, adjaceny_matrix, norm_A, config.realizations_n, config.realizations_bounds,
                config.la_m_depth, fake_news_data, merged_mis_timestamps, mitigated_users_per_stage, sample_size, sens_param)
    la_mhp_results = n.run(4)
    consumed_la_budget = sum(la_mhp_results.values())
    for user_id, _ in enumerate(norm_MU):
        norm_MU[user_id] += la_mhp_results[user_id]

    return norm_MU, consumed_la_budget