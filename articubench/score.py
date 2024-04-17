"""
Here the benchmark subscores and the total score are calculated.

"""
from collections import abc
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import euclidean_distances

from .util import speak, librosa_melspec, normalize_mel_librosa, get_vel_acc_jerk, RMSE, mel_to_tensor, cps_to_ema
from .eval_tongue_height import tongue_height_from_cps
from .embedding_models import MelEmbeddingModel
from .control_models import synth_baseline_schwa
from . import control_models
from . import benchmark_data

tqdm.pandas()

DIR = os.path.dirname(__file__)


"""
MAX_VEL_GECO = np.array([0.078452  , 0.0784    , 0.081156  , 0.05490857, 0.044404  ,
       0.016517  , 0.08116   , 0.04426   , 0.03542   , 0.04072   ,
       0.06958   , 0.03959091, 0.02612571, 0.02720448, 0.03931667,
       0.03434   , 0.024108  , 0.080998  , 0.081097  , 0.00109286,
       0.081634  , 0.02085943, 0.01849143, 0.0081164 , 0.        ,
       0.040585  , 0.0006106 , 0.        , 0.        , 0.        ])
"""
MAX_VEL_GECO = 0.081634
MAX_VEL_GECO = 150.0

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
MAX_JERK_GECO = 150.0

#LABEL_VECTORS = pd.read_pickle(os.path.join(DIR, "data/label_vectors.pkl"))
LABEL_VECTORS = pd.read_pickle(os.path.join(DIR, "data/lexical_embedding_vectors.pkl"))
LABEL_VECTORS_NP = np.array(list(LABEL_VECTORS.vector))

BASELINE_TONGUE_HEIGHT = None
BASELINE_EMA = None
BASELINE_SPECTROGRAM = None
BASELINE_LOUDNESS = None
BASELINE_SEMDIST = None


EMAS_TB = ['TONGUE_225-x[cm]', 'TONGUE_225-y[cm]', 'TONGUE_225-z[cm]']
EMAS_TT = ['TONGUE_115-x[cm]', 'TONGUE_115-y[cm]', 'TONGUE_115-z[cm]']

def _no_data_in_column(column, data):
    return ((column not in data.columns)
            or (data[column].isnull().any())
            or (not len(data.index)))


def score(model, *, preloaded_data=None, precomputed_scores=None, size='tiny', tasks=('copy-synthesis', 'semantic-acoustic',
    'semantic-only'), subscores='all', return_individual_subscores=False, device=torch.device('cpu')):
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

    start_time = time.time()

    if (not isinstance(subscores, abc.Sequence) and not isinstance(tasks, str)) or (isinstance(tasks, str) and tasks != 'all'):
        raise ValueError('tasks has to be either list of tasks or "all"')
    if (not isinstance(subscores, abc.Sequence) and not isinstance(subscores, str)) or (isinstance(subscores, str) and subscores != 'all'):
        raise ValueError('subscores has to be either list of subscores or "all"')

    if tasks == 'all':
        tasks = ('copy-synthesis', 'semantic-acoustic', 'semantic-only')
    if subscores == 'all':
        subscores = ('acoustic', 'articulatory', 'semantic')

    assert any([task in ('copy-synthesis', 'semantic-acoustic', 'semantic-only') for task in tasks]), "tasks has to be one of 'copy-synthesis', 'semantic-acoustic', 'semantic-only'"

    # load data
    print("load data")
    if preloaded_data is None:
        if size == 'tiny':
            data = benchmark_data.load_tiny()
        elif size == 'small':
            data = benchmark_data.load_small()
        elif size == 'normal':
            data = benchmark_data.load_normal()
        else:
            raise ValueError(f"size has to be one of 'tiny', 'small', 'normal' and not {size}")

    else:
        data = preloaded_data.copy()


    # generate Baseline:
    print("generate baseline")
    if _no_data_in_column('cps_baseline', data):
        data['cps_baseline'] = data.len_cp.progress_apply(synth_baseline_schwa)
    if _no_data_in_column('sig_baseline', data):
        data['sig_baseline'] = data['cps_baseline'].progress_apply(lambda cps: speak(cps)[0])

    # generate cps
    print("generate cps")
    if 'copy-synthesis' in tasks:
        if _no_data_in_column('cps_copy-synthesis', data):
            data['cps_copy-synthesis'] = data.progress_apply(
                    lambda row: model(row['len_cp'],
                                      target_audio=row['target_sig'],
                                      sampling_rate=row['target_sr']),
                    axis=1)
    if 'semantic-acoustic' in tasks:
        if _no_data_in_column('cps_semantic-acoustic', data):
            data['cps_semantic-acoustic'] = data.progress_apply(
                    lambda row: model(row['len_cp'],
                                      target_audio=row['target_sig'],
                                      sampling_rate=row['target_sr'],
                                      target_semantic_vector=row['target_semantic_vector']),
                    axis=1)
    if 'semantic-only' in tasks:
        if _no_data_in_column('cps_semantic-only', data):
            data['cps_semantic-only'] = data.progress_apply(
                    lambda row: model(row['len_cp'],
                                      target_semantic_vector=row['target_semantic_vector']),
                    axis=1)

    # synthesize cps
    print("synthesize cps")
    for task in tasks:
        if _no_data_in_column(f'sig_{task}', data):
            data[f'sig_{task}'] = data[f'cps_{task}'].progress_apply(lambda cps: speak(cps)[0]) # inv_normalize ?


    # calculate tongue heights
    print("calculate tongue height")
    if subscores == 'all' or 'articulatory' in subscores:
        if _no_data_in_column('tongue_height_baseline', data):
            data['tongue_height_baseline'] = data['cps_baseline'].progress_apply(lambda cps: tongue_height_from_cps(cps))
        global BASELINE_TONGUE_HEIGHT
        BASELINE_TONGUE_HEIGHT = np.mean(data.progress_apply(lambda row: RMSE(row['tongue_height_baseline'], row['reference_tongue_height']),axis=1))

        for task in tasks:
            if _no_data_in_column(f'tongue_height_{task}', data):
                data[f'tongue_height_{task}'] = data[f'cps_{task}'].progress_apply(lambda cps: tongue_height_from_cps(cps))
                # TODO: only calculate tongue height where ultra sound data is available for comparison TODO


    # calculate EMA 
    print("calculating EMAs")
    if subscores == 'all' or 'articulatory' in subscores:

        if _no_data_in_column('ema_TT_baseline', data):
                data['ema_TT_baseline'] = data['cps_baseline'].progress_apply(lambda cps: cps_to_ema(cps)[EMAS_TT].to_numpy())
        if _no_data_in_column('ema_TB_baseline', data):
                data['ema_TB_baseline'] = data['cps_baseline'].progress_apply(lambda cps: cps_to_ema(cps)[EMAS_TB].to_numpy())

        global BASELINE_EMA
        BASELINE_EMA = np.mean(data.progress_apply(lambda row: RMSE(row['ema_TT_baseline'], row['reference_ema_TT']), axis=1)
                        + data.progress_apply(lambda row: RMSE(row['ema_TB_baseline'], row['reference_ema_TB']), axis=1))

        for task in tasks:
            #we calculate Tongue Tip EMA
            if _no_data_in_column(f'ema_TT_{task}', data):
                data[f'ema_TT_{task}'] = data[f'cps_{task}'].progress_apply(lambda cps: cps_to_ema(cps)[EMAS_TT].to_numpy())
        
            #we calculate Tongue Body EMA
            if _no_data_in_column(f'ema_TB_{task}', data):
                data[f'ema_TB_{task}'] = data[f'cps_{task}'].progress_apply(lambda cps: cps_to_ema(cps)[EMAS_TB].to_numpy())
    
    # calculate log-mel spectrograms
    print("calculate log-mel spectrogram")
    if 'acoustic' in subscores or 'semantic' in subscores:
        if _no_data_in_column('log_mel_baseline', data):
                data['log_mel_baseline'] = data['sig_baseline'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig, 44100)))
        if _no_data_in_column('loudness_baseline', data):
                data['loudness_baseline'] = data['log_mel_baseline'].progress_apply(lambda x: np.mean(x, axis=1))

        for task in tasks:
            if _no_data_in_column(f'log_mel_{task}', data):
                data[f'log_mel_{task}'] = data[f'sig_{task}'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig,44100)))
            if _no_data_in_column(f'loudness_{task}', data):
                data[f'loudness_{task}'] = data[f'log_mel_{task}'].progress_apply(lambda x: np.mean(x, axis=1))


        if _no_data_in_column('target_log_mel', data):
            data['target_log_mel'] = data.progress_apply(lambda row: normalize_mel_librosa(librosa_melspec(row['target_sig'],row['target_sr'])),axis=1)
        if _no_data_in_column('target_loudness', data):
            data['target_loudness'] = data['target_log_mel'].progress_apply(lambda x: np.mean(x, axis=1))

        global BASELINE_SPECTROGRAM
        global BASELINE_LOUDNESS
        BASELINE_SPECTROGRAM = np.mean(data.progress_apply(lambda row: RMSE(row['log_mel_baseline'], row['target_log_mel']), axis=1))
        BASELINE_LOUDNESS = np.mean(data.progress_apply(lambda row: RMSE(row['loudness_baseline'], row['target_loudness']), axis=1))

    # predict vector embeddings
    print("predict vector embedding")
    if 'semantic' in subscores:
        embedder = MelEmbeddingModel(num_lstm_layers=2, hidden_size=720, dropout=0.7).double()
        embedder.load_state_dict(torch.load(
            os.path.join(DIR, "models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
            map_location=device))
        embedder = embedder.to(device)
        embedder.eval()

        with torch.no_grad():
            if _no_data_in_column('semantic_vector_baseline', data):
                data['semantic_vector_baseline'] = data['log_mel_baseline'].progress_apply(lambda mel: embedder(mel_to_tensor(mel), (torch.tensor(mel.shape[0]),))[-1,:].detach().cpu().numpy().copy())
            if _no_data_in_column('semantic_rank_baseline', data):
                data['semantic_rank_baseline'] = data.progress_apply(lambda row: sem_rank(row['semantic_vector_baseline'], row['label']),axis=1)

            global BASELINE_SEMDIST
            BASELINE_SEMDIST = np.mean(data.progress_apply(lambda row: RMSE(row['semantic_vector_baseline'], row['target_semantic_vector']),axis=1))

            for task in tasks:
                if _no_data_in_column(f'semantic_vector_{task}', data):
                    data[f'semantic_vector_{task}'] = data[f'log_mel_{task}'].progress_apply(lambda mel: embedder(mel_to_tensor(mel),(torch.tensor(mel.shape[0]),))[-1, :].detach().cpu().numpy().copy())
                if _no_data_in_column(f'semantic_rank_{task}', data):
                    data[f'semantic_rank_{task}'] = data.progress_apply(lambda row: sem_rank(row[f'semantic_vector_{task}'], row['label']),axis=1)


    # ????
    # refactor model from function to class with: model.copy_synthesis,
    # model.semantic_acoustic, model.semantic_only, model.name
    # ????

    #scores = dict()
    if precomputed_scores is None:
        scores = pd.DataFrame(columns = ['task', 'score_total'])
    else:
        scores = precomputed_scores.copy()

    scores['task'] = tasks

    print("Start tasks...")
    for task in tasks:
        # articulatory
        if 'articulatory' in subscores:
            print("articulatory task")
            if ('score_articulatory' not in scores.columns) or (scores.loc[scores.task == task, 'score_articulatory'].isnull().any()) or (not len(scores.index)):
                s_articulatory, subscores_articulatory = score_articulatory(data, task=task)
                if return_individual_subscores:
                    scores.loc[scores.task == task, 'score_articulatory/tongue_height'] = subscores_articulatory[0]
                    scores.loc[scores.task == task, 'score_articulatory/ema'] = subscores_articulatory[1]
                    scores.loc[scores.task == task, 'score_articulatory/velocity_jerk'] = subscores_articulatory[2]
                else:
                    scores.loc[scores.task == task, 'score_articulatory'] = s_articulatory

        # acoustic
        if 'acoustic' in subscores:
            print("acoustic task")
            if ('score_acoustic' not in scores.columns) or (scores.loc[scores.task == task, 'score_acoustic'].isnull().any()) or (not len(scores.index)):
                s_acoustic, subscores_acoustic = score_acoustic(data, task=task)
                if return_individual_subscores:
                    scores.loc[scores.task == task, 'score_acoustic/loudness'] = subscores_acoustic[0]
                    scores.loc[scores.task == task, 'score_acoustic/spectrogram'] = subscores_acoustic[1]
                else:
                    scores.loc[scores.task == task, 'score_acoustic'] = s_acoustic

        # semantic
        if 'semantic' in subscores:
            print("semantic task")
            if ('score_semantic' not in scores.columns) or (scores.loc[scores.task == task, 'score_semantic'].isnull().any()) or (not len(scores.index)):
                s_semantic, subscores_semantic = score_semantic(data, task=task)
                if return_individual_subscores:
                    scores.loc[scores.task == task, 'score_semantic/distance'] = subscores_semantic[0]
                    scores.loc[scores.task == task, 'score_semantic/rank'] = subscores_semantic[1]
                else:
                    scores.loc[scores.task == task, 'score_semantic'] = s_semantic

    scores["score_total"] = scores.iloc[:, 2:].sum(axis=1)

    minutes = (time.time() - start_time) / 60
    print(f"TOTAL WALL TIME USED: {minutes:.2f} min")

    return scores.score_total, scores, data


########################################
########## Score Articulatory ##########
########################################
def score_articulatory(data, *, task):
    s_tongue_height = score_tongue_height(data, task=task)
    s_ema = score_ema(data, task=task)
    s_vel_jerk = score_vel_jerk(data, task=task)

    s_articulatory = s_tongue_height + s_ema + s_vel_jerk
    return s_articulatory, [s_tongue_height, s_ema, s_vel_jerk]


def score_tongue_height(data, task):
    s_tongue_height = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'tongue_height_{task}'], row['reference_tongue_height']), axis=1)) / BASELINE_TONGUE_HEIGHT)
    return s_tongue_height


def score_ema(data, task):
    s_ema = 100 * (1 - (np.mean(data.progress_apply(lambda row: RMSE(row[f'ema_TT_{task}'], row['reference_ema_TT']), axis=1)
                        + data.progress_apply(lambda row: RMSE(row[f'ema_TB_{task}'], row['reference_ema_TB']), axis=1))
                        / BASELINE_EMA))
    return s_ema


def score_vel_jerk(data, task):
    """
    On a logarithmic scale and cannot be lower than 0, even for very bad
    (high) velocities and jerks.

    Use the 99.9% quantile instead of the maximal value to be a little bit
    better prepared against very few outliers.

    Note: Uses side effects to add columns to data inplace.

    """
    temp_dat = list(data[f'cps_{task}'].apply(get_vel_acc_jerk))
    df_temp = pd.DataFrame(temp_dat, columns=['vel', 'acc', 'jerk'])
    max_vels = df_temp.vel.apply(np.amax)
    max_jerks = df_temp.jerk.apply(np.amax)
    quantile_vels = df_temp.vel.apply(lambda x: np.quantile(x, .999))
    quantile_jerks = df_temp.jerk.apply(lambda x: np.quantile(x, .999))

    data[f'max_vel_{task}'] = max_vels
    data[f'max_jerk_{task}'] = max_jerks
    data[f'quantile999_vel_{task}'] = quantile_vels
    data[f'quantile999_jerk_{task}'] = quantile_jerks

    del df_temp
    del temp_dat

    #s_vel = 1 - min(np.mean(np.log(1 + quantile_vels)) / np.log(1 + MAX_VEL_GECO), 0.5)
    #print(f's_vel: {s_vel}')
    #s_jerk = 1 - min(np.mean(np.log(1 + quantile_jerks)) / np.log(1 + MAX_JERK_GECO), 0.5)
    #print(f's_jerk: {s_jerk}')

    s_vel_jerk = 100 * (1 - min(np.mean(np.log(1 + quantile_vels)) / np.log(1 + MAX_VEL_GECO), 0.5)
                          - min(np.mean(np.log(1 + quantile_jerks)) / np.log(1 + MAX_JERK_GECO), 0.5))
    return s_vel_jerk


########################################
############ Score acoustic ############
########################################
def score_acoustic(data, *, task):
    s_loudness = score_loudness(data,task=task)
    s_spectrogram = score_spectrogram(data, task=task)

    s_acoustic = s_loudness + s_spectrogram
    return s_acoustic, [s_loudness, s_spectrogram]

def score_loudness(data,task):
    s_loudness = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'loudness_{task}'],row['target_loudness']),axis=1))/ BASELINE_LOUDNESS)
    return s_loudness

def score_spectrogram(data, task):
    s_spectrogram = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'log_mel_{task}'], row['target_log_mel']),axis=1)) / BASELINE_SPECTROGRAM)
    return s_spectrogram


########################################
############ Score Semantic ############
########################################

def score_semantic(data, *, task):
    s_sem_dist = score_sem_dist(data,task=task)
    s_sem_rank = score_sem_rank(data, task=task)

    s_semantic = s_sem_dist + s_sem_rank
    return s_semantic, [s_sem_dist, s_sem_rank]

def score_sem_dist(data, task):
    s_sem_dist = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'semantic_vector_{task}'], row['target_semantic_vector']),axis=1)) / BASELINE_SEMDIST)
    return s_sem_dist

def score_sem_rank(data,task):
    s_sem_rank = 100*(1-np.mean(data[f'semantic_rank_{task}']-1)/4311)
    return s_sem_rank

def sem_rank(semvec, label):
    pred_semvec = np.asarray([semvec])
    dist = euclidean_distances(pred_semvec, LABEL_VECTORS_NP)[0]
    true_index = int(LABEL_VECTORS[LABEL_VECTORS.label == label].index[0])
    #dist_target = dist[true_index]
    dist_argsort = np.argsort(dist)
    rank_target = np.where(dist_argsort == true_index)[0][0] + 1
    return rank_target


    results = score(control_models.synth_paule_fast)

