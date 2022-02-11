"""
Here the benchmark subscores and the total score are calculated.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
"""
MAX_VEL_GECO = np.array([0.078452  , 0.0784    , 0.081156  , 0.05490857, 0.044404  ,
       0.016517  , 0.08116   , 0.04426   , 0.03542   , 0.04072   ,
       0.06958   , 0.03959091, 0.02612571, 0.02720448, 0.03931667,
       0.03434   , 0.024108  , 0.080998  , 0.081097  , 0.00109286,
       0.081634  , 0.02085943, 0.01849143, 0.0081164 , 0.        ,
       0.040585  , 0.0006106 , 0.        , 0.        , 0.        ])
"""
MAX_VEL_GECO = 0.081634
"""
MAX_JERK_GECO = np.array([1.83420000e-02, 3.17840000e-02, 2.38400000e-03, 1.78314286e-02,
       2.47940000e-02, 1.23300000e-03, 3.47480000e-02, 9.73363636e-03,
       1.22457143e-02, 1.46800000e-02, 1.86000000e-03, 5.59890909e-03,
       5.68000000e-03, 5.50247750e-03, 1.30963333e-02, 1.18733333e-02,
       7.16000000e-04, 1.53860000e-02, 9.56100000e-03, 1.07142857e-05,
       6.85760060e-03, 3.40171429e-03, 3.00342857e-03, 1.32360000e-03,
       0.00000000e+00, 1.28800000e-03, 2.22000000e-05, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00])
"""
MAX_JERK_GECO = 0.034748



def score(model, *, size='tiny', tasks=('copy-synthesis', 'semantic-acoustic',
    'semantic-only'), subscores='all'):
    """
    Main function to calculate the score for the benchmark.

    Parameters
    ==========
    model : control model
        function that takes a target semantic vector and / or target signal as
        input and returns a control parameter trajectory for the vtl.
    size : {'tiny', 'small', 'normal'}
        the size of the benchmark data to evaluate against
    tasks : list or 'all'
        list of tasks to evaluate
    subscores : list or 'all'
        list of which subscores to calculate

    Returns
    =======
    score, subscores : dict(subscore: score)

    """

    # load data
    if size == 'tiny':
        data = benchmark_data.load_tiny()
    elif size == 'small':
        data = benchmark_data.load_small()
    elif size == 'normal':
        data = benchmark_data.load_normal()
    else:
        raise ValueError(f"size has to be one of 'tiny', 'small', 'normal' and not {size}")

    # generate cps
    if tasks == 'all' or 'copy-synthesis' in tasks:
        data['cps_copy-synthesis'] = data.progress_apply(lambda row: model(row['len_cp'], target_audio=row['rec_sig'], sampling_rate=row['rec_sr']))
    if tasks == 'all' or 'semantic-acoustic' in tasks:
        data['cps_semantic-acoustic'] = data.progress_apply(lambda row: model(row['len_cp'], target_audio=row['rec_sig'], sampling_rate=row['rec_sr'], target_semantic_vector=row['target_semantic_vector']))
    if tasks == 'all' or 'semantic-only' in tasks:
        data['cps_semantic-only'] = data.progress_apply(lambda row: model(row['len_cp'], target_semantic_vector=row['target_semantic_vector']))

    # synthesise cps
    if tasks == 'all' or 'copy-synthesis' in tasks:
        data['sig_copy-synthesis'] = data['cps_copy-synthesis'].progress_apply(lambda cps: util.speak(cps)[0])
    if tasks == 'all' or 'semantic-acoustic' in tasks:
        data['sig_semantic-acoustic'] = data['cps_semantic-acoustic'].progress_apply(lambda cps: util.speak(cps)[0])
    if tasks == 'all' or 'semantic-only' in tasks:
        data['sig_semantic-only'] = data['cps_semantic-only'].progress_apply(lambda cps: util.speak(cps)[0])

    # ????
    # refactor model from function to class with: model.copy_synthesis,
    # model.semantic_acoustic, model.semantic_only, model.name
    # ????

    #scores = dict()
    scores = pd.DataFrame(columns = ['task', 'score_total', 'score_acoustic','score_articulatory', 'score_semantic'])
    if tasks == 'all':
        tasks = ('copy-synthesis', 'semantic-acoustic',
    'semantic-only')

    scores['task'] = tasks

    for task in tasks:
        # articulation
        if subscores == 'all' or 'articulatory' in subscores:
            if scores.loc[scores.task == task, f'score_articulation'].isnull().any():
                s_articulatory, subscores_articulatory = score_articulatory(data, task=task)
                scores.loc[scores.task == task, f'score_articulation'] = s_articulation

        # acoustic
        if subscores == 'all' or 'acoustic' in subscores:
            if scores.loc[scores.task == task, f'score_acoustic'].isnull().any():
                s_acoustic, subscores_acoustic = score_acoustic(data, tasks=tasks)
                scores.loc[scores.task == task, f'score_articulation'] = s_acoustic

        # semantic
        if subscores == 'all' or 'semantic' in subscores:
            if scores.loc[scores.task == task, f'score_semantic'].isnull().any():
                s_semantic, subscores_semantic = score_semantic(data, tasks=tasks)
                scores.loc[scores.task == task, f'score_semantic'] = s_acoustic


    return scores.score_total, scores


def score_articulatory(data, *, task):
    s_tongue_height = score_tongue_height(data,task=task)
    s_ema = score_ema(data, task=task)
    s_vel_jerk = score_vel_jerk(data,task=task)

    s_articulatory = s_tongue_height + s_ema + s_vel_jerk
    return s_articulatory, [s_tongue_height, s_ema, s_vel_jerk]





def score_vel_jerk(data,task):
    temp_dat = data[f'cps_{task}'].apply(util.get_vel_acc_jerk)
    df_temp = pd.DataFrame(temp_dat, columns=['vel', 'acc', 'jerk'])
    max_vels = df_temp.vel.apply(np.amax)
    max_jerks = df_temp.jerk.apply(np.amax)

    #data[f'max_vel_{task}'] = max_vels
    #data[f'max_jerk_{task}'] = max_jerks

    s_vel_jerk = 100 * (2 - np.mean(max_vels) / MAX_VEL_GECO - np.mean(max_jerks) / MAX_JERK_GECO)
    return s_vel_jerk