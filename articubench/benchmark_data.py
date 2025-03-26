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

from .eval_tongue_height import tongue_height_from_cps
from .util import cps_to_ema
from tqdm import tqdm
tqdm.pandas()
EMAS_TB = ['TONGUE_225-x[cm]', 'TONGUE_225-y[cm]', 'TONGUE_225-z[cm]']
EMAS_TT = ['TONGUE_115-x[cm]', 'TONGUE_115-y[cm]', 'TONGUE_115-z[cm]']

DIR = os.path.dirname(__file__)


def load_tiny():
    data = pd.read_pickle(os.path.join(DIR, 'data/tiny.pkl'))

    if 'reference_cp' not in data.columns:
        data['reference_cp'] = None
    if 'reference_tongue_height' not in data.columns:
        data['reference_tongue_height'] = None
    if 'reference_ema_TT' not in data.columns:
        data['reference_ema_TT'] = None
    if 'reference_ema_TB' not in data.columns:
        data['reference_ema_TB'] = None

    # recompute len_cp
    data['len_cp'] = data.apply(lambda row: int(2 * math.ceil(44100 / 220 * len(row['target_sig']) / row['target_sr'])), axis=1)
    tongue_heights = data.reference_cp.apply(lambda cp: tongue_height_from_cps(cp) if cp is not None else None)
    emas_tt = data.reference_cp.apply(lambda cp: cps_to_ema(cp)[EMAS_TT].to_numpy() if cp is not None else None)
    emas_tb = data.reference_cp.apply(lambda cp: cps_to_ema(cp)[EMAS_TB].to_numpy() if cp is not None else None)
    data.loc[~emas_tt.isna(), 'reference_ema_TT'] = emas_tt[~emas_tt.isna()]
    data.loc[~emas_tb.isna(), 'reference_ema_TB'] = emas_tb[~emas_tb.isna()]
    data.loc[~tongue_heights.isna(), 'reference_tongue_height'] = tongue_heights[~tongue_heights.isna()]
    return data


def load_small():
    data = pd.read_pickle(os.path.join(DIR, 'data/small.pkl'))
    if 'reference_cp' not in data.columns:
        data['reference_cp'] = None
    if 'reference_tongue_height' not in data.columns:
        data['reference_tongue_height'] = None
    if 'reference_ema_TT' not in data.columns:
        data['reference_ema_TT'] = None
    if 'reference_ema_TB' not in data.columns:
        data['reference_ema_TB'] = None


    #we dont compute lengths since i precomputed them already

    #we drop rows which have a length of a multiple of 32, since librosa melspec gives us spectograms which are 1 too long, which results in wrong length spectrograms in acoustic condition
    data.drop(data[data['len_cp'] % 32 == 0].index, inplace=True)
    #we also drop ref_emas for noow, since i dont know why they have the wrong shape
    #data['reference_ema_TT'] = None
    #data['reference_ema_TB'] = None
    tongue_heights = data.reference_cp.progress_apply(lambda cp: tongue_height_from_cps(cp) if cp is not None else None)
    data['emas'] = data.reference_cp.progress_apply(lambda cp: cps_to_ema(cp) if cp is not None else None)
    emas_tt = data['emas'].progress_apply(lambda emas: emas[EMAS_TT].to_numpy() if emas is not None else None)
    emas_tb = data['emas'].progress_apply(lambda emas: emas[EMAS_TB].to_numpy() if emas is not None else None)
    data.loc[~emas_tt.isna(), 'reference_ema_TT'] = emas_tt[~emas_tt.isna()]
    data.loc[~emas_tb.isna(), 'reference_ema_TB'] = emas_tb[~emas_tb.isna()]
    data.loc[~tongue_heights.isna(), 'reference_tongue_height'] = tongue_heights[~tongue_heights.isna()]
    return data

def load_normal():
    data = None
    return data

