"""
Here the benchmark subscores and the total score are calculated.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os

from util import speak, librosa_melspec, normalize_mel_librosa, get_vel_acc_jerk, RMSE, mel_to_tensor
from eval_tongue_height import tongue_heights_from_cps
from embedding_models import MelEmbeddingModel
from  control_models import synth_baseline_schwa

tqdm.pandas()
from sklearn.metrics.pairwise import euclidean_distances




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

LABEL_VECTORS = pd.read_pickle("/data/label_vectors_vectors_checked.pkl")
LABEL_VECTORS_NP = np.array(list(LABEL_VECTORS.vector))

def score(model, *, data=None, scores=None, size='tiny', tasks=('copy-synthesis', 'semantic-acoustic',
    'semantic-only'), subscores='all', device = torch.device('cpu')):
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
    if data is None:
        if size == 'tiny':
            data = benchmark_data.load_tiny()
        elif size == 'small':
            data = benchmark_data.load_small()
        elif size == 'normal':
            data = benchmark_data.load_normal()
        else:
            raise ValueError(f"size has to be one of 'tiny', 'small', 'normal' and not {size}")

    # generate Baseline:
    if ('cps_baseline' not in data.columns) or (data['cps_baseline'].isnull().any()) or (not len(data.index)):
        data['cps_baseline'] = data.len_cp.progress_apply(synth_baseline_schwa)
    if ('sig_baseline' not in data.columns) or (data['sig_baseline'].isnull().any()) or (not len(data.index)):
        data['sig_baseline'] = data['cps_baseline'].progress_apply(lambda cps: speak(cps)[0])

    # generate cps
    if tasks == 'all' or 'copy-synthesis' in tasks:
        if ('cps_copy-synthesis' not in data.columns) or (data['cps_copy-synthesis'].isnull().any()) or (not len(data.index)):
                data['cps_copy-synthesis'] = data.progress_apply(lambda row: model(row['len_cp'], target_audio=row['target_sig'], sampling_rate=row['target_sr']))

    if tasks == 'all' or 'semantic-acoustic' in tasks:
        if ('cps_semantic-acoustic' not in data.columns) or (data['cps_semantic-acoustic'].isnull().any()) or (not len(data.index)):
            data['cps_semantic-acoustic'] = data.progress_apply(lambda row: model(row['len_cp'], target_audio=row['target_sig'], sampling_rate=row['target_sr'], target_semantic_vector=row['target_semantic_vector']))

    if tasks == 'all' or 'semantic-only' in tasks:
        if ('cps_semantic-only' not in data.columns) or (data['cps_semantic-only'].isnull().any()) or (not len(data.index)):
            data['cps_semantic-only'] = data.progress_apply(lambda row: model(row['len_cp'], target_semantic_vector=row['target_semantic_vector']))

    # synthesise cps
    if tasks == 'all' or 'copy-synthesis' in tasks:
        if ('sig_copy-synthesis' not in data.columns) or (data['sig_copy-synthesis'].isnull().any()) or (not len(data.index)):
            data['sig_copy-synthesis'] = data['cps_copy-synthesis'].progress_apply(lambda cps: speak(cps)[0]) # inv_normalize ?

    if tasks == 'all' or 'semantic-acoustic' in tasks:
        if ('sig_semantic-acoustic' not in data.columns) or (data['sig_semantic-acoustic'].isnull().any()) or (not len(data.index)):
            data['sig_semantic-acoustic'] = data['cps_semantic-acoustic'].progress_apply(lambda cps: speak(cps)[0])

    if tasks == 'all' or 'semantic-only' in tasks:
        if ('sig_semantic-only' not in data.columns) or (data['sig_semantic-only'].isnull().any()) or (not len(data.index)):
            data['sig_semantic-only'] = data['cps_semantic-only'].progress_apply(lambda cps: speak(cps)[0])

    # calculate tongue heights
    if subscores == 'all' or 'articulatory' in subscores:
        if ('tongue_heights_baseline' not in data.columns) or (data['tongue_heights_baseline'].isnull().any()) or (not len(data.index)):
            data['tongue_heights_baseline'] = data['cps_baseline'].progress_apply(lambda cps: tongue_heights_from_cps(cps))
        global BASELINE_TONGUE_HEIGHT
        BASELINE_TONGUE_HEIGHT = np.mean(data.progress_apply(lambda row: RMSE(row['tongue_heights_baseline'], row['tongue_heights_ultra'])))

        if tasks == 'all' or 'copy-synthesis' in tasks:
            if ('tongue_heights_copy-synthesis' not in data.columns) or (data['tongue_heights_copy-synthesis'].isnull().any()) or (not len(data.index)):
                data['tongue_heights_copy-synthesis'] = data['cps_copy-synthesis'].progress_apply(lambda cps: tongue_heights_from_cps(cps))

        if tasks == 'all' or 'semantic-acoustic' in tasks:
            if ('tongue_heights_semantic-acoustic' not in data.columns) or (data['tongue_heights_semantic-acoustic'].isnull().any()) or (not len(data.index)):
                data['tongue_heights_semantic-acoustic'] = data['cps_semantic-acoustic'].progress_apply(lambda cps: tongue_heights_from_cps(cps))

        if tasks == 'all' or 'semantic-only' in tasks:
            if ('tongue_heights_semantic-only' not in data.columns) or (data['tongue_heights_semantic-only'].isnull().any()) or (not len(data.index)):
                data['tongue_heights_semantic-only'] = data['cps_semantic-only'].progress_apply(lambda cps: tongue_heights_from_cps(cps))


    # calculate log-mel spectrograms
    if subscores == 'all' or 'acoustic' in subscores or 'semantic' in subscores:
        if ('log_mel_baseline' not in data.columns) or (data['log_mel_baseline'].isnull().any()) or (not len(data.index)):
                data['log_mel_baseline'] = data['sig_baseline'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig, 44100)))
                data['loudness_baseline'] = data['log_mel_baseline'].progress_apply(lambda x: np.mean(x, axis=1))

        global BASELINE_SPECTROGRAM
        global BASELINE_LOUDNESS
        BASELINE_SPECTROGRAM = np.mean(data.progress_apply(lambda row: RMSE(row[f'log_mel_{task}'], row['target_log_mel'])))
        BASELINE_LOUDNESS = np.mean(data.progress_apply(lambda row: RMSE(row[f'loudness_{task}'], row['target_loudness'])))

        if tasks == 'all' or 'copy-synthesis' in tasks:
            if ('log_mel_copy-synthesis' not in data.columns) or (data['log_mel_copy-synthesis'].isnull().any()) or (not len(data.index)):
                data['log_mel_copy-synthesis'] = data['sig_copy-synthesis'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig,44100)))
                data['loudness_copy-synthesis'] = data['log_mel_copy-synthesis'].progress_apply(lambda x: np.mean(x, axis=1))

        if tasks == 'all' or 'semantic-acoustic' in tasks:
            if ('log_mel_semantic-acoustic' not in data.columns) or (data['log_mel_semantic-acoustic'].isnull().any()) or (not len(data.index)):
                data['log_mel_semantic-acoustic'] = data['sig_semantic-acoustic'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig,44100)))
                data['loudness_semantic-acoustic'] = data['log_mel_semantic-acoustic'].progress_apply(lambda x: np.mean(x, axis=1))

        if tasks == 'all' or 'semantic-only' in tasks:
            if ('log_mel_semantic-only' not in data.columns) or (data['log_mel_semantic-only'].isnull().any()) or (not len(data.index)):
                data['log_mel_semantic-only'] = data['sig_semantic-only'].progress_apply(lambda sig: normalize_mel_librosa(librosa_melspec(sig,44100)))
                data['loudness_semantic-only'] = data['log_mel_semantic-only'].progress_apply(lambda x: np.mean(x, axis=1))

        if ('target_log_mel' not in data.columns) or (data['target_log_mel'].isnull().any()) or (not len(data.index)):
            data['target_log_mel'] = data.progress_apply(lambda row: normalize_mel_librosa(librosa_melspec(row['target_sig'],row['target_sr'])))
            data['target_loudness'] = data['target_log_mel'].progress_apply(lambda x: np.mean(x, axis=1))

    # predict vector embeddings
    if subscores == 'all' or 'semantic' in subscores:
        embedder = MelEmbeddingModel(num_lstm_layers=2, hidden_size=720, dropout=0.7).double()
        embedder.load_state_dict(torch.load(
            os.path.join("../pretrained_models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
            map_location=device))
        embedder = embedder.to(device)
        embedder.eval()

        with torch.no_grad:
            if ('semantic_vector_baseline' not in data.columns) or (data['semantic_vector_baseline'].isnull().any()) or (not len(data.index)):
                data['semantic_vector_baseline'] = data['log_mel_baseline'].progress_apply(lambda mel: embedder(mel_to_tensor(mel), (torch.tensor(mel.shape[0])))[-1,:].detach().cpu().numpy().copy())
                data['semantic_rank_baseline'] = data.progress_apply(lambda row: sem_rank(row['semantic_vector_baseline'], row['label']))
            global BASELINE_SEMDIST
            BASELINE_SEMDIST = np.mean(data.progress_apply(lambda row: RMSE(row['semantic_vector_baseline'], row['target_semantic_vector'])))

            if tasks == 'all' or 'copy-synthesis' in tasks:
                if ('semantic_vector_copy-synthesis' not in data.columns) or (data['semantic_vector_copy-synthesis'].isnull().any()) or (not len(data.index)):
                    data['semantic_vector_copy-synthesis'] = data['log_mel_copy-synthesis'].progress_apply(lambda mel: embedder(mel_to_tensor(mel),(torch.tensor(mel.shape[0])))[-1, :].detach().cpu().numpy().copy())
                    data['semantic_rank_copy-synthesis'] = data.progress_apply(lambda row: sem_rank(row['semantic_vector_copy-synthesis'], row['label']))

            if tasks == 'all' or 'semantic-acoustic' in tasks:
                if ('semantic_vector_semantic-acoustic' not in data.columns) or (data['semantic_vector_semantic-acoustic'].isnull().any()) or (not len(data.index)):
                    data['semantic_vector_semantic-acoustic'] = data['log_mel_semantic-acoustic'].progress_apply(lambda mel: embedder(mel_to_tensor(mel),(torch.tensor(mel.shape[0])))[-1, :].detach().cpu().numpy().copy())
                    data['semantic_rank_semantic-acoustic'] = data.progress_apply(lambda row: sem_rank(row['semantic_vector_semantic-acoustic'], row['label']))


            if tasks == 'all' or 'semantic-only' in tasks:
                if ('semantic_vector_semantic-only' not in data.columns) or (data['semantic_vector_semantic-only'].isnull().any()) or (not len(data.index)):
                    data['semantic_vector_semantic-only'] = data['log_mel_semantic-only'].progress_apply(lambda mel: embedder(mel_to_tensor(mel),(torch.tensor(mel.shape[0])))[-1, :].detach().cpu().numpy().copy())
                    data['semantic_rank_semantic-only'] = data.progress_apply(lambda row: sem_rank(row['semantic_vector_semantic-only'], row['label']))


    # ????
    # refactor model from function to class with: model.copy_synthesis,
    # model.semantic_acoustic, model.semantic_only, model.name
    # ????

    #scores = dict()
    if scores is None:
        scores = pd.DataFrame(columns = ['task', 'score_total', 'score_acoustic','score_articulatory', 'score_semantic'])

    if tasks == 'all':
        tasks = ('copy-synthesis', 'semantic-acoustic',
    'semantic-only')

    scores['task'] = tasks

    for task in tasks:
        # articulation
        if subscores == 'all' or 'articulatory' in subscores:
            if ('score_articulatory' not in scores.columns) or (scores.loc[scores.task == task, 'score_articulatory'].isnull().any()) or (not len(scores.index)):
                s_articulatory, subscores_articulatory = score_articulatory(data, task=task)
                scores.loc[scores.task == task, 'score_articulation'] = s_articulatory

        # acoustic
        if subscores == 'all' or 'acoustic' in subscores:
            if ('score_acoustic' not in scores.columns) or (scores.loc[scores.task == task, 'score_acoustic'].isnull().any()) or (not len(scores.index)):
                s_acoustic, subscores_acoustic = score_acoustic(data, tasks=tasks)
                scores.loc[scores.task == task, 'score_articulation'] = s_acoustic

        # semantic
        if subscores == 'all' or 'semantic' in subscores:
            if ('score_semantic' not in scores.columns) or (scores.loc[scores.task == task, 'score_semantic'].isnull().any()) or (not len(scores.index)):
                s_semantic, subscores_semantic = score_semantic(data, tasks=tasks)
                scores.loc[scores.task == task, 'score_semantic'] = s_semantic

    scores["score_total"] = scores.iloc[:,2:].sum(axis=1)/3

    return scores.score_total, scores


########################################
########## Score Articulatory ##########
########################################
def score_articulatory(data, *, task):
    s_tongue_height = score_tongue_height(data,task=task)
    s_ema = score_ema(data, task=task)
    s_vel_jerk = score_vel_jerk(data,task=task)

    s_articulatory = s_tongue_height + s_ema + s_vel_jerk
    return s_articulatory, [s_tongue_height, s_ema, s_vel_jerk]

def score_tongue_height(data, task):
    s_tongue_height = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'tongue_heights_{task}'], row['tongue_heights_ultra']))) / BASELINE_TONGUE_HEIGHT)
    return s_tongue_height

def score_ema(data, task):


def score_vel_jerk(data,task):
    temp_dat = data[f'cps_{task}'].apply(get_vel_acc_jerk)
    df_temp = pd.DataFrame(temp_dat, columns=['vel', 'acc', 'jerk'])
    max_vels = df_temp.vel.apply(np.amax)
    max_jerks = df_temp.jerk.apply(np.amax)

    data[f'max_vel_{task}'] = max_vels
    data[f'max_jerk_{task}'] = max_jerks

    del df_temp
    del temp_dat

    s_vel_jerk = 100 * (2 - np.mean(max_vels) / MAX_VEL_GECO - np.mean(max_jerks) / MAX_JERK_GECO)
    return s_vel_jerk



########################################
############ Score acoustic ############
########################################
def score_acoustic(data, *, task):
    s_loudness = score_loudness(data,task=task)
    s_sepctrogram = score_spectrogram(data, task=task)

    s_acoustic = s_loudness + s_sepctrogram
    return s_acoustic, [s_loudness, s_sepctrogram]

def score_loudness(data,task):
    s_loudness = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'loudness_{task}'],row['target_loudness'])))/ BASELINE_LOUDNESS)
    return s_loudness

def score_spectrogram(data, task):
    s_spectrogram = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'log_mel_{task}'], row['target_log_mel']))) / BASELINE_SPECTROGRAM)
    return s_spectrogram


########################################
############ Score Semantic ############
########################################

def score_semantic(data, *, task):
    s_sem_dist = score_sem_dist(data,task=task)
    s_sem_rank = score_sem_rank(data, task=task)

    s_semantic = s_sem_dist + s_sem_rank
    return s_semantic, [s_sem_dist, s_sem_rank]

def score_sem_dist(data,task):
    s_sem_dist = 100 * (1 - np.mean(data.progress_apply(lambda row: RMSE(row[f'semantic_vector_{task}'], row['target_semantic_vector']))) / BASELINE_SEMDIST)
    return s_sem_dist

def score_sem_rank(data,task):
    s_sem_rank = 100*(1-np.mean(data[f'semantic_rank_{task}']-1)/4311)
    return s_sem_rank

def sem_rank(pred_semvec, label):
    dist = euclidean_distances(pred_semvec, LABEL_VECTORS_NP)[0]
    true_index = int(LABEL_VECTORS[LABEL_VECTORS.label == label].index[0])
    #dist_target = dist[true_index]
    dist_argsort = np.argsort(dist)
    rank_target = np.where(dist_argsort == true_index)[0][0] + 1
    return rank_target