"""
Here the differently sized benchmark datasets are defined.

The datasets use a wide format, where initially only the the semvec and the
audio and the duration are defined per row; each score, task, control model
combination adds a column to all rows with the results.

"""

import os
import math

import pandas as pd
import numpy as np


DIR = os.path.dirname(__file__)


def load_tiny():
    data = pd.read_pickle(os.path.join(DIR, 'data/geco_tiny.pkl'))
    data['target_sig'] = data['rec_sig']
    data['target_sr'] = data['rec_sr']
    data['target_semantic_vector'] = data['vector']
    data['tongue_heights_ultra'] = [np.array([np.NaN]) for _ in range(data.shape[0])]
    if 'len_cp' not in data.columns:
        data['len_cp'] = data.apply(lambda row: int(2 * math.ceil(44100 / 220 * len(row['target_sig']) / row['target_sr'])), axis=1)
    del data['cp_norm']
    return data


def load_small():
    data = None
    return data


def load_normal():
    data = None
    return data

