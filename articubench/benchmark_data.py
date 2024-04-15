"""
Here the differently sized benchmark datasets are defined.

The datasets use a wide format, where initially only the the semvec and the
audio and the duration are defined per row; each score, task, control model
combination adds a column to all rows with the results.

The DataFrames come with the following columns:

    file : str
        The source file name which consitutes a unique identifier and allows
        for tracking the sample back to its source.
    label : str
        Word type / class of the sample
    target_semantic_vector : np.array
        300 dimensional fasttext vector of the label
    target_sig : np.array
        variable lenght mono signal of the sample
    target_sr : int
        sampling rate of the target_sig signal
    len_cp : int
        length of the control parameters that should be generated for this
        sample. The length can be calculated the following way:
        int(2 * math.ceil(44100 / 220 * len(row['target_sig']) / row['target_sr']))
    reference_cp : np.array or None
        if the target_sig contains a synthesis by the VocalTractLab these are
        the control parameter that lead to that target_sig signal. The
        reference_cp for this sample lead to the (simulated) EMA points and the
        (simulated) tongue height for this sample.
    reference_tongue_height : np.array or None
        shape = (seq_length); seq_length is with approx 80.2 Hz (1 : 550/44100)
        the tongue height either measured from a human with ultra sound or
        simulated from a reference_cp or None if no ultra sound recording is
        available for the human recording.
    reference_ema : np.array or None
        shape = (4, seq_length); seq_length is with ???? Hz
        the tongue tip (x, y) and tongue middle / body (x, y) sensor data of an
        EMA recording or simulated EMA points from a reference_cp or None if no
        EMA recording was conducted for the human recording.

"""

import os
import math

import pandas as pd
import numpy as np

from .eval_tongue_height import tongue_height_from_cps
from .util import cps_to_ema

EMAS = ['TONGUE_115-x[cm]', 'TONGUE_115-y[cm]', 'TONGUE_115-z[cm]','TONGUE_225-x[cm]', 'TONGUE_225-y[cm]', 'TONGUE_225-z[cm]', 'TONGUE_335-x[cm]', 'TONGUE_335-y[cm]', 'TONGUE_335-z[cm]']
 

DIR = os.path.dirname(__file__)


def load_tiny():
    data = pd.read_pickle(os.path.join(DIR, 'data/tiny.pkl'))

    if 'reference_cp' not in data.columns:
        data['reference_cp'] = None
    if 'reference_tongue_height' not in data.columns:
        data['reference_tongue_height'] = None
    if 'reference_ema' not in data.columns:
        data['reference_ema'] = None

    # recompute len_cp
    data['len_cp'] = data.apply(lambda row: int(2 * math.ceil(44100 / 220 * len(row['target_sig']) / row['target_sr'])), axis=1)
    tongue_heights = data.reference_cp.apply(lambda cp: tongue_height_from_cps(cp) if cp is not None else None)
    emas = data.reference_cp.apply(lambda cp: cps_to_ema(cp)[EMAS].to_numpy() if cp is not None else None)
    data['reference_ema'][~emas.isna()] = emas[~emas.isna()]
    data['reference_tongue_height'][~tongue_heights.isna()] = tongue_heights[~tongue_heights.isna()]
    return data


def load_small():
    data = None
    return data


def load_normal():
    data = None
    return data

