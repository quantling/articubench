"""
Here the benchmark subscores and the total score are calculated.

"""

from collections import abc
import os
import time
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import euclidean_distances

from .util import (
    speak,
    librosa_melspec,
    normalize_mel_librosa,
    get_vel_acc_jerk,
    RMSE,
    mel_to_tensor,
    cps_to_ema,
    align_ema,
    DIR,
    DEVICE,
    CPU_CORES,
    MAX_VEL_GECO,
    MAX_JERK_GECO,
    LABEL_VECTORS,
    LABEL_VECTORS_NP,
    BASELINE_TONGUE_HEIGHT,
    BASELINE_EMA,
    BASELINE_SPECTROGRAM,
    BASELINE_LOUDNESS,
    BASELINE_SEMDIST,
    EMAS_TB,
    EMAS_TT
)

from .eval_tongue_height import tongue_height_from_cps
from .embedding_models import MelEmbeddingModel
from .control_models import synth_baseline_schwa
from . import benchmark_data


tqdm.pandas()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _no_data_in_column(column, data):
    return (
        (column not in data.columns)
        or (data[column].isnull().any())
        or (not len(data.index))
    )


def process_task(task, data_subset):
    if _no_data_in_column(f"sig_{task}", data_subset):
        # data[f'sig_{task}'] = data[f'cps_{task}'].progress_apply(lambda cps: speak(cps)[0])
        data_subset[f"sig_{task}"] = data_subset[f"cps_{task}"].progress_apply(
            lambda cps: speak(cps)[0]
        )


def apply_speak(cps):
    return speak(cps)[0]


def apply_tongue_height_from_cps(cps):
    return tongue_height_from_cps(cps)


def score(
    model,
    *,
    preloaded_data=None,
    precomputed_scores=None,
    size="tiny",
    tasks="all",
    subscores="all",
    return_individual_subscores=False
):
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

    if (not isinstance(subscores, abc.Sequence) and not isinstance(tasks, (list, tuple, str))) or (
        isinstance(tasks, str) and tasks != "all"
    ):
        raise ValueError('tasks has to be either list of tasks or "all"')
    if (not isinstance(subscores, abc.Sequence) and not isinstance(subscores, str)) or (
        isinstance(subscores, str) and subscores != "all"
    ):
        raise ValueError('subscores has to be either list of subscores or "all"')

    if tasks == "all":
        tasks = ("copy-synthesis", "semantic-acoustic", "semantic-only")
    if subscores == "all":
        subscores = ("acoustic", "articulatory", "semantic")

    tasks = ("baseline",) + tasks

    assert any(
        [
            task in ("copy-synthesis", "semantic-acoustic", "semantic-only")
            for task in tasks
        ]
    ), "tasks has to be one of 'copy-synthesis', 'semantic-acoustic', 'semantic-only'"

    # load data
    print("load data")
    if preloaded_data is None:
        if size == "tiny":
            data = benchmark_data.load_tiny()
        elif size == "small":
            data = benchmark_data.load_small()
        elif size == "normal":
            data = benchmark_data.load_normal()
        else:
            raise ValueError(
                f"size has to be one of 'tiny', 'small', 'normal' and not {size}"
            )

    else:
        data = preloaded_data.copy()

    if _no_data_in_column("cps_baseline", data):
        logging.info("Calculating baseline cps")
        data["cps_baseline"] = data.len_cp.progress_apply(synth_baseline_schwa)

    if "copy-synthesis" in tasks:
        if _no_data_in_column("cps_copy-synthesis", data):
            logging.info("Calculating copy-synthesis cps")
            data["cps_copy-synthesis"] = data.progress_apply(
                lambda row: model(
                    row["len_cp"],
                    target_audio=row["target_sig"],
                    sampling_rate=row["target_sr"],
                ),
                axis=1,
            )

    if "semantic-acoustic" in tasks:
        if _no_data_in_column("cps_semantic-acoustic", data):
            logging.info("Calculating semantic-acoustic cps")
            data["cps_semantic-acoustic"] = data.progress_apply(
                lambda row: model(
                    row["len_cp"],
                    target_audio=row["target_sig"],
                    sampling_rate=row["target_sr"],
                    target_semantic_vector=row["target_semantic_vector"],
                ),
                axis=1,
            )

    if "semantic-only" in tasks:
        if _no_data_in_column("cps_semantic-only", data):
            logging.info("Calculating semantic-only cps")
            data["cps_semantic-only"] = data.progress_apply(
                lambda row: model(
                    row["len_cp"], target_semantic_vector=row["target_semantic_vector"]
                ),
                axis=1,
            )

    for task in tasks:
        logging.info(f"synthesizing {task} cps")
        if _no_data_in_column(f"sig_{task}", data):
            with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
                results = list(
                    tqdm(
                        executor.map(apply_speak, data[f"cps_{task}"]), total=len(data)
                    )
                )
                data[f"sig_{task}"] = results

    # calculate tongue heights
    print("calculate tongue height")
    if subscores == "all" or "articulatory" in subscores:
        for task in tasks:
            logging.info(f"calculating tongue height of {task} task")

            if _no_data_in_column(f"tongue_height_{task}", data):
                with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
                    results = list(
                        tqdm(
                            executor.map(tongue_height_from_cps, data[f"cps_{task}"]),
                            total=len(data),
                        )
                    )
                    data[f"tongue_height_{task}"] = results

                # TODO: only calculate tongue height where ultra sound data is available for comparison TODO

        global BASELINE_TONGUE_HEIGHT
        BASELINE_TONGUE_HEIGHT = np.mean(
            data.progress_apply(
                lambda row: RMSE(
                    row["tongue_height_baseline"], row["reference_tongue_height"]
                ),
                axis=1,
            )
        )

        for task in tasks:
            logging.info(f"Calculating EMA Points of {task} task")

            with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
                results = list(
                    tqdm(executor.map(cps_to_ema, data[f"cps_{task}"]), total=len(data))
                )
                data[f"ema_points_{task}"] = results

            if _no_data_in_column(f"ema_TT_{task}", data):
                data[f"ema_TT_{task}"] = data[f"ema_points_{task}"].progress_apply(
                    lambda emas: emas[EMAS_TT].to_numpy()
                )

            # we calculate Tongue Body EMA
            if _no_data_in_column(f"ema_TB_{task}", data):
                data[f"ema_TB_{task}"] = data[f"ema_points_{task}"].progress_apply(
                    lambda emas: emas[EMAS_TB].to_numpy()
                )

        global BASELINE_EMA
        BASELINE_EMA = np.mean(
            data.progress_apply(
                lambda row: RMSE(
                    *align_ema(
                        row["ema_TT_baseline"], row["reference_ema_TT"]
                        )
                ),
                axis=1,
            )
            + data.progress_apply(
                lambda row: RMSE(
                    *align_ema(
                        row["ema_TB_baseline"], row["reference_ema_TB"]
                        )
                ),
                axis=1,
            )
        )

    # calculate log-mel spectrograms
    print("calculate log-mel spectrogram")
    if "acoustic" in subscores or "semantic" in subscores:
        for task in tasks:
            logging.info(f"calculating log-mel spectrogram of {task} task")
            if _no_data_in_column(f"log_mel_{task}", data):
                data[f"log_mel_{task}"] = data[f"sig_{task}"].progress_apply(
                    lambda sig: normalize_mel_librosa(librosa_melspec(sig, 44100))
                )

            logging.info(f"calculating loudness of {task} task")
            if _no_data_in_column(f"loudness_{task}", data):
                data[f"loudness_{task}"] = data[f"log_mel_{task}"].progress_apply(
                    lambda x: np.mean(x, axis=1)
                )

        if _no_data_in_column("target_log_mel", data):
            data["target_log_mel"] = data.progress_apply(
                lambda row: normalize_mel_librosa(
                    librosa_melspec(row["target_sig"], row["target_sr"])
                ),
                axis=1,
            )
        if _no_data_in_column("target_loudness", data):
            data["target_loudness"] = data["target_log_mel"].progress_apply(
                lambda x: np.mean(x, axis=1)
            )

        global BASELINE_SPECTROGRAM
        global BASELINE_LOUDNESS
        BASELINE_SPECTROGRAM = np.mean(
            data.progress_apply(
                lambda row: RMSE(row["log_mel_baseline"], row["target_log_mel"]), axis=1
            )
        )
        BASELINE_LOUDNESS = np.mean(
            data.progress_apply(
                lambda row: RMSE(row["loudness_baseline"], row["target_loudness"]),
                axis=1,
            )
        )

    # predict vector embeddings
    print("predict vector embedding")
    if "semantic" in subscores:
        embedder = MelEmbeddingModel(
            num_lstm_layers=2, hidden_size=720, dropout=0.7
        ).double()
        embedder.load_state_dict(
            torch.load(
                os.path.join(
                    DIR,
                    "models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
                map_location=DEVICE,
                weights_only=True
            )
        )
        embedder = embedder.to(DEVICE)
        embedder.eval()
        with torch.no_grad():
            for task in tasks:
                if _no_data_in_column(f"semantic_vector_{task}", data):
                    data[f"semantic_vector_{task}"] = data[
                        f"log_mel_{task}"
                    ].progress_apply(
                        lambda mel: embedder(
                            mel_to_tensor(mel), (torch.tensor(mel.shape[0]),)
                        )[-1, :]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                if _no_data_in_column(f"semantic_rank_{task}", data):
                    data[f"semantic_rank_{task}"] = data.progress_apply(
                        lambda row: sem_rank(
                            row[f"semantic_vector_{task}"], row["label"]
                        ),
                        axis=1,
                    )

            global BASELINE_SEMDIST
            BASELINE_SEMDIST = np.mean(
                data.progress_apply(
                    lambda row: RMSE(
                        row["semantic_vector_baseline"], row["target_semantic_vector"]
                    ),
                    axis=1,
                )
            )

    # ????
    # refactor model from function to class with: model.copy_synthesis,
    # model.semantic_acoustic, model.semantic_only, model.name
    # ????

    # scores = dict()
    if precomputed_scores is None:
        scores = pd.DataFrame(columns=["task", "score_total"])
    else:
        scores = precomputed_scores.copy()

    scores["task"] = tasks

    print("Start tasks...")
    for task in tasks:
        # articulatory
        if "articulatory" in subscores:
            print("articulatory task")
            if _no_data_in_column("scores_articulatory", scores):
                s_articulatory, subscores_articulatory = score_articulatory(
                    data, task=task
                )
                if return_individual_subscores:
                    scores.loc[
                        scores.task == task, "score_articulatory/tongue_height"
                    ] = subscores_articulatory[0]
                    scores.loc[scores.task == task, "score_articulatory/ema"] = (
                        subscores_articulatory[1]
                    )
                    scores.loc[
                        scores.task == task, "score_articulatory/velocity_jerk"
                    ] = subscores_articulatory[2]
                else:
                    scores.loc[scores.task == task, "score_articulatory"] = (
                        s_articulatory
                    )

        # acoustic
        if "acoustic" in subscores:
            print("acoustic task")
            if _no_data_in_column("score_acoustic", scores):
                s_acoustic, subscores_acoustic = score_acoustic(data, task=task)
                if return_individual_subscores:
                    scores.loc[scores.task == task, "score_acoustic/loudness"] = (
                        subscores_acoustic[0]
                    )
                    scores.loc[scores.task == task, "score_acoustic/spectrogram"] = (
                        subscores_acoustic[1]
                    )
                else:
                    scores.loc[scores.task == task, "score_acoustic"] = s_acoustic

        # semantic
        if "semantic" in subscores:
            print("semantic task")
            if _no_data_in_column("score_semantic", scores):
                s_semantic, subscores_semantic = score_semantic(data, task=task)
                if return_individual_subscores:
                    scores.loc[scores.task == task, "score_semantic/distance"] = (
                        subscores_semantic[0]
                    )
                    scores.loc[scores.task == task, "score_semantic/rank"] = (
                        subscores_semantic[1]
                    )
                else:
                    scores.loc[scores.task == task, "score_semantic"] = s_semantic

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
    s_tongue_height = 100 * (
        1
        - np.mean(
            data.progress_apply(
                lambda row: RMSE(
                    row[f"tongue_height_{task}"], row["reference_tongue_height"]
                ),
                axis=1,
            )
        )
        / BASELINE_TONGUE_HEIGHT
    )
    return s_tongue_height


def score_ema(data, task):
    s_ema = 100 * (
        1
        - np.mean(
            data.progress_apply(
                lambda row: RMSE(
                    *align_ema(row[f"ema_TT_{task}"], row["reference_ema_TT"])
                ),
                axis=1,
            )
            + data.progress_apply(
                lambda row: RMSE(
                    *align_ema(row[f"ema_TB_{task}"], row["reference_ema_TB"])
                ),
                axis=1,
            )
        )
        / BASELINE_EMA
    )
    return s_ema


def score_vel_jerk(data, task):
    """
    On a logarithmic scale and cannot be lower than 0, even for very bad
    (high) velocities and jerks.

    Use the 99.9% quantile instead of the maximal value to be a little bit
    better prepared against very few outliers.

    Note: Uses side effects to add columns to data inplace.

    """
    temp_dat = list(data[f"cps_{task}"].apply(get_vel_acc_jerk))
    df_temp = pd.DataFrame(temp_dat, columns=["vel", "acc", "jerk"])
    max_vels = df_temp.vel.apply(np.amax)
    max_jerks = df_temp.jerk.apply(np.amax)
    quantile_vels = df_temp.vel.apply(lambda x: np.quantile(x, 0.999))
    quantile_jerks = df_temp.jerk.apply(lambda x: np.quantile(x, 0.999))

    data[f"max_vel_{task}"] = max_vels
    data[f"max_jerk_{task}"] = max_jerks
    data[f"quantile999_vel_{task}"] = quantile_vels
    data[f"quantile999_jerk_{task}"] = quantile_jerks

    del df_temp
    del temp_dat

    # s_vel = 1 - min(np.mean(np.log(1 + quantile_vels)) / np.log(1 + MAX_VEL_GECO), 0.5)
    # print(f's_vel: {s_vel}')
    # s_jerk = 1 - min(np.mean(np.log(1 + quantile_jerks)) / np.log(1 + MAX_JERK_GECO), 0.5)
    # print(f's_jerk: {s_jerk}')

    s_vel_jerk = 100 * (
        1
        - min(np.mean(np.log(1 + quantile_vels)) / np.log(1 + MAX_VEL_GECO), 0.5)
        - min(np.mean(np.log(1 + quantile_jerks)) / np.log(1 + MAX_JERK_GECO), 0.5)
    )
    return s_vel_jerk


########################################
############ Score acoustic ############
########################################
def score_acoustic(data, *, task):
    s_loudness = score_loudness(data, task=task)
    s_spectrogram = score_spectrogram(data, task=task)

    s_acoustic = s_loudness + s_spectrogram
    return s_acoustic, [s_loudness, s_spectrogram]


def score_loudness(data, task):
    s_loudness = 100 * (
        1
        - np.mean(
            data.progress_apply(
                lambda row: RMSE(row[f"loudness_{task}"], row["target_loudness"]),
                axis=1,
            )
        )
        / BASELINE_LOUDNESS
    )
    return s_loudness


def score_spectrogram(data, task):
    s_spectrogram = 100 * (
        1
        - np.mean(
            data.progress_apply(
                lambda row: RMSE(row[f"log_mel_{task}"], row["target_log_mel"]), axis=1
            )
        )
        / BASELINE_SPECTROGRAM
    )
    return s_spectrogram


########################################
############ Score Semantic ############
########################################


def score_semantic(data, *, task):
    s_sem_dist = score_sem_dist(data, task=task)
    s_sem_rank = score_sem_rank(data, task=task)

    s_semantic = s_sem_dist + s_sem_rank
    return s_semantic, [s_sem_dist, s_sem_rank]


def score_sem_dist(data, task):
    s_sem_dist = 100 * (
        1
        - np.mean(
            data.progress_apply(
                lambda row: RMSE(
                    row[f"semantic_vector_{task}"], row["target_semantic_vector"]
                ),
                axis=1,
            )
        )
        / BASELINE_SEMDIST
    )
    return s_sem_dist


def score_sem_rank(data, task):
    s_sem_rank = 100 * (1 - np.mean(data[f"semantic_rank_{task}"] - 1) / 4311)
    return s_sem_rank


def sem_rank(semvec, label):
    pred_semvec = np.asarray([semvec])
    dist = euclidean_distances(pred_semvec, LABEL_VECTORS_NP)[0]
    true_index = int(LABEL_VECTORS[LABEL_VECTORS.label == label].index[0])
    # dist_target = dist[true_index]
    dist_argsort = np.argsort(dist)
    rank_target = np.where(dist_argsort == true_index)[0][0] + 1
    return rank_target
