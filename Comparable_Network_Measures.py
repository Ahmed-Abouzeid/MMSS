######################################################
#  This module is used from the work in              #
#   Abouzeid, et, al 2021 to compare with            #
######################################################
from numpy.core.defchararray import isnumeric


def calc_correlation(fake_news_avg_exposures, true_news_avg_exposures):
    """function to calculate the impact correlation between fake and true (before and after mitigation) news as
    each stage

    Parameters
    -----------------
    fake_news_avg_exposures: list
                list of averaged values on n stages
    true_news_avg_exposures: list
                list of averaged values on n stages

    Returns
    --------------
    list of correlations at each stage"""

    stages_correlations = []
    for e, stage_fake_exposure in enumerate(fake_news_avg_exposures):
        stages_correlations.append(true_news_avg_exposures[e] * stage_fake_exposure)
    return stages_correlations


def calc_difference(fake_news_avg_exposures, true_news_avg_exposures):
    """function to calculate the impact differences between fake and true (before and after mitigation) news as
        each stage

    Parameters
    -----------------
    fake_news_avg_exposures: list
                list of averaged values on n stages
    true_news_avg_exposures: list
                list of averaged values on n stages

    Returns
    --------------
    list of differences at each stage"""

    stages_differences = []
    for e, stage_fake_exposure in enumerate(fake_news_avg_exposures):
        stages_differences.append(stage_fake_exposure - true_news_avg_exposures[e])
    return stages_differences


def get_selected_users_per_stage(fake_news_most_exposed_users):
    """function to return only the ids of users per stage where they had more exposures to fake news compared to true news

    Parameters
    ---------------------
    fake_news_most_exposed_users: list
                a list of n stages lists, each include a list of triples for each user (only the higher exposed to fake news)
                 exposures in that stage
    Returns
    --------------------
    list
        list of n stages where each stage list is a list of user ids that had ight exposures to fake news in that list

    """
    stages_user_ids = []
    for stage_id, _ in enumerate(fake_news_most_exposed_users):
        stage_user_ids = []
        for result in fake_news_most_exposed_users[stage_id]:
            user_id = result[0]
            stage_user_ids.append(user_id)
        stages_user_ids.append(stage_user_ids)

    return stages_user_ids

def chunk_timestamps(timestamps, chunks_bounds, convert_value = None):
    """function to transform timestamps into chunks per hour, in order to be able to evaluate simulation results per
    each time stage (hour)

    Parameters
    -----------------------
    timestamps: list
                a list of numpy array each represents a user timestamps
    chunks_bounds: list
                a list of maximum time each time stage should have as a chunk, the given boundaries should be ordered
                as by each hour boundary (3600 in seconds) to second hour boundary (7200 in seconds) and so on
    convert_value: int, optional
                a value used to convert the time value from a simulated stamps into same scale as test stamps, since the
                first hour or second or so on from the simulation will be compared to the the 11th, 12th, ... hours
                from the test stamps
    Returns
    --------------------
    list
        a list of lists, representing all chunks (lists) per each user"""

    if type(timestamps) is dict:
        timestamps = list(timestamps.values())

    users_timestamps_per_2hour = []
    for user_stamps in timestamps:
        user_chunks = []
        for e, bound in enumerate(chunks_bounds):
            chunk = []
            for t in user_stamps:
                if convert_value:
                    t -= convert_value
                if e * chunks_bounds[0] < t <= bound:
                    chunk.append(t)

            user_chunks.append(chunk)

        users_timestamps_per_2hour.append(user_chunks)

    return users_timestamps_per_2hour



def get_user_influencer(user_index, adjacency):
    """function to get a list of a user influencers from the adjacency matrix passed
    Parameters
    ----------------------
    user_index: int
                the id of the user to investigate
    adjacency: numpy array (n_users X n_users)
                the matrix from where we get the value of influence between any two users

    Returns
    ----------------------
    python set
            all the unique users ids who influence the argued user index"""

    influencers_indices = set()
    for e, col in enumerate(adjacency[user_index]):
        if adjacency[user_index][e] > 0:
            influencers_indices.add(e)
    return influencers_indices



def calc_user_campaign_exposure(user_index, adjacency, timestamps_chunks, stages_count=None, stage_id=None):
    """function to calculate the on a user level how he- she is affected by a campaign event by counting and weighting
    influencer users contribution in the same campaign

    Parameters
    ----------------------
    user_index: int
                the user index to calculate for
    adjacency:  numpy array (n_users X n_users)
                the matrix from where we get the value of influence between any two users
    timestamps_chunks: list
                    a list of n stages other lists which by turn includes n users timestamps
    stages_count: int, optional
                in case, given, it means that the required exposures will be on all stages
                , then the given limit,  helps to loop over all stages to get exposure per stage for the targeted user.
                if not given, means that the calculation is done per only one stage.

    stage_id: int
        if stages_count was not given, stage_id should be given to retrieve the certain stage_id information
    Returns
    ---------------------
    list
        a list of n stages lists, which by turn has the one user list which by turn has one value per each user indicating
        the impact of the campaign on that user"""

    if stages_count != None and stage_id == None:
        stages_impacts = []
        for s in range(stages_count):
            stage_impact = 0
            for user_ind in get_user_influencer(user_index, adjacency):
                if adjacency[user_index][user_ind] > 0:
                    for old_stage in range(s):
                        stage_impact += len(timestamps_chunks[user_ind][old_stage])
                    stage_impact += len(timestamps_chunks[user_ind][s])
            stages_impacts.append(stage_impact)

        return stages_impacts

    elif stages_count == None and stage_id != None:
        stage_impact = 0
        for user_ind in get_user_influencer(user_index, adjacency):
            if adjacency[user_index][user_ind] > 0:
                for old_stage in range(stage_id):
                    stage_impact += len(timestamps_chunks[user_ind][old_stage])
                stage_impact += len(timestamps_chunks[user_ind][stage_id])
            #stage_impact += len(timestamps_chunks[user_ind][stage_id]) * adjacency[user_index][user_ind]

        return stage_impact


def get_stages_avg_exposures(stages_users_exposures):
    """function to calculate the average impact of fake news at network users per time stage (realization), it does also the same
    for the impact of true news before and after the mitigation. It uses the counts of shared events a subset of users did
    and it corresponds that to the user who is influenced by these users, such information of influence is obtain from
    the passed adjacency matrix. The function only gets results for a targetted users as those with higher exposure to fake news
    before the mitigation. that is to make the results more sensational and focused on how the results for those users change after mitigation

    Parameters
    ------------------
    stages_users_exposures: list
                n stages lists where each stage list contains the high exposures as triple where user_id, true_news_exposure,
                and fake news exposure

    Returns
    -----------------------
    list
        a list of n stages length where each element is an averaged true news exposure on a stage calculated from all users exposures in
        that stage
    list
        a list of n stages length where each element is an averaged fake news exposure on a stage calculated from all users exposures in
        that stage
    """
    stages_true_news_avg_exposures = []
    stages_fake_news_avg_exposures = []
    for stage_id, stage in enumerate(stages_users_exposures):
        stage_true_news_exposures = []
        stage_fake_news_exposures = []
        for user_exposure in stage:
            stage_true_news_exposures.append(user_exposure[1])
            stage_fake_news_exposures.append(user_exposure[2])
        if len(stage) != 0:
            stages_true_news_avg_exposures.append(sum(stage_true_news_exposures)/ len(stage))
            stages_fake_news_avg_exposures.append(sum(stage_fake_news_exposures)/ len(stage))
        else:
            stages_true_news_avg_exposures.append(0)
            stages_fake_news_avg_exposures.append(0)

    return stages_true_news_avg_exposures, stages_fake_news_avg_exposures

def get_high_exposures_users(fake_timestamps, true_stamps, network_adjacency, stages_count, bounds, mitigated_users):
    """function to receive all experiment users timestamps (counts of events) of both fake and true news campaign events,
      either before and after then it select only those with higher exposures (impact on them) to fake news campaign events
      (stage_based where each stage might have different users)

    Parameters
    ---------------------
    fake_timestamps: list of numpy array
                  a list of numpy arrays, each numpy array is a user fake news sharing timestamps, and they are
                  chunked per stage for all users
    true_stamps: list of numpy array
                  a list of numpy arrays, each numpy array is a user true news before or after mitigation and sharing
                   timestamps, and they are chunked per stage for all users
    network_adjacency: numpy array
                    a matrix where influence information will be used for users when collecting impacts of event
    stages_count: int
                    used to loop through the stages
    mitigated_users: list
                  a list of n stages list, each contains list of user ids in this list, it is used when calculating exposures
                  after mitigation

    Returns
    --------------------
    list
        a list of n stages list of triples, each triple is a a user id, true news (before or after mitigation)
         exposure value at that stage, and fake news exposure at same stage"""

    #stages_bounds = [x * 600 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
    #stages_bounds = [x * 3600 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]

    #stages_bounds = [3600, 7200, 10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000]
    stages_bounds = bounds

    fake_news_chunks = chunk_timestamps(fake_timestamps, stages_bounds, convert_value=None)
    true_news_chunks = chunk_timestamps(true_stamps, stages_bounds, convert_value=None)
    stages_most_exposed_users = []
    for stage_id in range(stages_count):
        stage_most_exposed_users = []
        for user_id, _ in enumerate(network_adjacency[0]):
            if mitigated_users == None:
                user_fake_exposure = calc_user_campaign_exposure(user_id, network_adjacency, fake_news_chunks, None, stage_id)
                user_true_exposure = calc_user_campaign_exposure(user_id, network_adjacency, true_news_chunks, None, stage_id)
                if user_fake_exposure > user_true_exposure:
                    stage_most_exposed_users.append((user_id, user_true_exposure, user_fake_exposure))
            else:
                if user_id in mitigated_users[stage_id]:
                    user_fake_exposure = calc_user_campaign_exposure(user_id, network_adjacency, fake_news_chunks, None,
                                                                        stage_id)
                    user_true_exposure = calc_user_campaign_exposure(user_id, network_adjacency,
                                                                     true_news_chunks, None, stage_id)
                    stage_most_exposed_users.append((user_id, user_true_exposure, user_fake_exposure))

        stages_most_exposed_users.append(stage_most_exposed_users)
    return stages_most_exposed_users
