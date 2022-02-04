"""
Here the benchmark subscores and the total score are calculated.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


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
        the size of the benchmark data to evlaute against
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
    elif tasks == 'all' or 'semantic-acoustic' in tasks:
        data['cps_semantic-acoustic'] = data.progress_apply(lambda row: model(row['len_cp'], target_audio=row['rec_sig'], sampling_rate=row['rec_sr'], target_semantic_vector=row['target_semantic_vector']))
    elif tasks == 'all' or 'semantic-only' in tasks:
        data['cps_semantic-only'] = data.progress_apply(lambda row: model(row['len_cp'], target_semantic_vector=row['target_semantic_vector']))

    # synthesise cps
    if tasks == 'all' or 'copy-synthesis' in tasks:
        data['sig_copy-synthesis'] = data['cps_copy-synthesis'].progress_apply(lambda cps: util.speak(cps)[0])
    elif tasks == 'all' or 'semantic-acoustic' in tasks:
        data['sig_semantic-acoustic'] = data['cps_semantic-acoustic'].progress_apply(lambda cps: util.speak(cps)[0])
    elif tasks == 'all' or 'semantic-only' in tasks:
        data['sig_semantic-only'] = data['cps_semantic-only'].progress_apply(lambda cps: util.speak(cps)[0])

    # ????
    # refactor model from function to class with: model.copy_synthesis,
    # model.semantic_acoustic, model.semantic_only, model.name
    # ????

    scores = dict()

    # articulation
    if subscores == 'all' or 'articulation' in subscores:
        s_articulation, subscores_articulation = score_articulation(model, size=size, tasks=tasks)
        scores['score_articulation'] = s_articulation

    # acoustic
    if subscores == 'all' or 'acoustic' in subscores:
        s_acoustic, subscores_acoustic = score_acoustic(model, size=size, tasks=tasks)
        scores['score_acoustic'] = s_acoustic

    # semantic
    if subscores == 'all' or 'semantic' in subscores:
        s_semantic, subscores_semantic = score_semantic(model, size=size, tasks=tasks)
        scores['score_semantic'] = s_semantic


    return np.mean(scores.values()), scores


def score_articulation(data, *, size, tasks='all'):
    scores = dict()

