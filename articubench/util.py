import ctypes
import math
from math import asin, pi, atan2, cos
import os
import sys
import tempfile
import zipfile
import io
import shutil

import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch.nn
import pandas as pd
import torch
import requests
from scipy.interpolate import PchipInterpolator

#setting some package constants
DIR = os.path.dirname(__file__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_CORES = 8
"""
MAX_VEL_GECO = np.array([0.078452  , 0.0784    , 0.081156  , 0.05490857, 0.044404  ,
       0.016517  , 0.08116   , 0.04426   , 0.03542   , 0.04072   ,
       0.06958   , 0.03959091, 0.02612571, 0.02720448, 0.03931667,
       0.03434   , 0.024108  , 0.080998  , 0.081097  , 0.00109286,
       0.081634  , 0.02085943, 0.01849143, 0.0081164 , 0.        ,
       0.040585  , 0.0006106 , 0.        , 0.        , 0.        ])
"""
MAX_VEL_GECO = 0.081634 #infered from GECO dataset
MAX_VEL_GECO = 150.0    #setted manually as our MAX

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
MAX_JERK_GECO = 0.034748 #infered from GECO dataset
MAX_JERK_GECO = 150.0    #setted manually as our MAX

LABEL_VECTORS = pd.read_pickle(os.path.join(DIR, "data/lexical_embedding_vectors.pkl"))

LABEL_VECTORS_NP = np.array(list(LABEL_VECTORS.vector))

BASELINE_TONGUE_HEIGHT = None
BASELINE_EMA = None
BASELINE_SPECTROGRAM = None
BASELINE_LOUDNESS = None
BASELINE_SEMDIST = None


EMAS_TB = ["TONGUE_225-x[cm]", "TONGUE_225-y[cm]", "TONGUE_225-z[cm]"]
EMAS_TT = ["TONGUE_115-x[cm]", "TONGUE_115-y[cm]", "TONGUE_115-z[cm]"]


# load vocaltractlab binary
PREFIX = "lib"
SUFFIX = ""
if sys.platform.startswith("linux"):
    SUFFIX = ".so"
elif sys.platform.startswith("win32"):
    PREFIX = ""
    SUFFIX = ".dll"
elif sys.platform.startswith("darwin"):
    SUFFIX = ".dylib"
VTL = ctypes.cdll.LoadLibrary(
    os.path.join(DIR, f"vocaltractlab_api/{PREFIX}VocalTractLabApi{SUFFIX}")
)
# initialize vtl
speaker_file_name = ctypes.c_char_p(
    os.path.join(DIR, "vocaltractlab_api/JD3.speaker").encode()
)
failure = VTL.vtlInitialize(speaker_file_name)
if failure != 0:
    raise ValueError("Error in vtlInitialize! Errorcode: %i" % failure)
del PREFIX, SUFFIX, speaker_file_name, failure
# get version / compile date
VERSION = ctypes.c_char_p(b" " * 64)
VTL.vtlGetVersion(VERSION)
print('Version of the VocalTractLab library: "%s"' % VERSION.value.decode())
del VERSION


# This should be done on all cp_deltas
# np.max(np.stack((np.abs(np.min(delta, axis=0)), np.max(delta, axis=0))), axis=0)
# np.max(np.stack((np.abs(np.min(cp_param, axis=0)), np.max(cp_param, axis=0))), axis=0)

# absolute value from max / min

# Vocal tract parameters: "HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3"
# Glottis parameters: "f0 pressure x_bottom x_top chink_area lag rel_amp double_pulsing pulse_skewness flutter aspiration_strength "


cp_means = np.array(
    [
        5.3000e-01,
        -5.0800e00,
        -3.0000e-02,
        -3.7300e00,
        7.0000e-02,
        7.3000e-01,
        4.8000e-01,
        -5.0000e-02,
        9.6000e-01,
        -1.5800e00,
        4.4600e00,
        -9.3000e-01,
        2.9900e00,
        -5.0000e-02,
        -1.4600e00,
        -2.2900e00,
        2.3000e-01,
        1.2000e-01,
        1.2000e-01,
        1.0720e02,
        4.1929e03,
        3.0000e-02,
        3.0000e-02,
        6.0000e-02,
        1.2200e00,
        8.4000e-01,
        5.0000e-02,
        0.0000e00,
        2.5000e01,
        -1.0000e01,
    ]
)
cp_stds = np.array(
    [
        1.70000e-01,
        4.00000e-01,
        4.00000e-02,
        6.30000e-01,
        1.20000e-01,
        2.20000e-01,
        2.20000e-01,
        9.00000e-02,
        4.90000e-01,
        3.10000e-01,
        3.80000e-01,
        3.70000e-01,
        3.50000e-01,
        3.50000e-01,
        4.60000e-01,
        3.80000e-01,
        6.00000e-02,
        1.00000e-01,
        1.80000e-01,
        9.86000e00,
        3.29025e03,
        2.00000e-02,
        2.00000e-02,
        1.00000e-02,
        0.00100e00,
        2.00000e-01,
        0.00100e00,
        0.00100e00,
        0.00100e00,
        0.00100e00,
    ]
)

cp_theoretical_means = np.array(
    [
        5.00000e-01,
        -4.75000e00,
        -2.50000e-01,
        -3.50000e00,
        0.00000e00,
        1.00000e00,
        5.00000e-01,
        4.50000e-01,
        5.00000e-01,
        -1.00000e00,
        3.50000e00,
        -2.50000e-01,
        5.00000e-01,
        1.00000e00,
        -1.00000e00,
        -3.00000e00,
        5.00000e-01,
        5.00000e-01,
        0.00000e00,
        3.20000e02,
        1.00000e04,
        1.25000e-01,
        1.25000e-01,
        0.00000e00,
        1.57075e00,
        0.00000e00,
        5.00000e-01,
        0.00000e00,
        5.00000e01,
        -2.00000e01,
    ]
)

cp_theoretical_stds = np.array(
    [
        5.00000e-01,
        1.25000e00,
        2.50000e-01,
        3.50000e00,
        1.00000e00,
        3.00000e00,
        5.00000e-01,
        5.50000e-01,
        3.50000e00,
        2.00000e00,
        2.00000e00,
        2.75000e00,
        3.50000e00,
        4.00000e00,
        3.00000e00,
        3.00000e00,
        5.00000e-01,
        5.00000e-01,
        1.00000e00,
        2.80000e02,
        1.00000e04,
        1.75000e-01,
        1.75000e-01,
        2.50000e-01,
        1.57075e00,
        1.00000e00,
        5.00000e-01,
        5.00000e-01,
        5.00000e01,
        2.00000e01,
    ]
)


def librosa_melspec(wav, sample_rate):
    wav = librosa.resample(
        wav,
        orig_sr=sample_rate,
        target_sr=44100,
        res_type="kaiser_best",
        fix=True,
        scale=False,
    )
    melspec = librosa.feature.melspectrogram(
        y=wav,
        n_fft=1024,
        hop_length=220,
        n_mels=60,
        sr=44100,
        power=1.0,
        fmin=10,
        fmax=12000,
    )
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order="C")


def normalize_cp(cp):
    return (cp - cp_theoretical_means) / cp_theoretical_stds


def inv_normalize_cp(norm_cp):
    return cp_theoretical_stds * norm_cp + cp_theoretical_means


# -83.52182518111363
mel_mean_librosa = librosa_melspec(np.zeros(5000), 44100)[0, 0]
mel_std_librosa = abs(mel_mean_librosa)


def normalize_mel_librosa(mel):
    return (mel - mel_mean_librosa) / mel_std_librosa


def inv_normalize_mel_librosa(norm_mel):
    return mel_std_librosa * norm_mel + mel_mean_librosa


def read_cp(filename):
    with open(filename, "rt") as cp_file:
        # skip first 6 lines
        for _ in range(6):
            cp_file.readline()
        glottis_model = cp_file.readline().strip()
        if glottis_model != "Geometric glottis":
            print(glottis_model)
            raise ValueError(
                f'glottis model is not "Geometric glottis" in file {filename}'
            )
        n_states = int(cp_file.readline().strip())
        cp_param = np.zeros((n_states, 19 + 11))
        for ii, line in enumerate(cp_file):
            kk = ii // 2
            if kk >= n_states:
                raise ValueError(
                    f"more states saved in file {filename} than claimed in the beginning"
                )
            # even numbers are glottis params
            elif ii % 2 == 0:
                glottis_param = line.strip()
                cp_param[kk, 19:30] = np.mat(glottis_param)
            # odd numbers are tract params
            elif ii % 2 == 1:
                tract_param = line.strip()
                cp_param[kk, 0:19] = np.mat(tract_param)
    return cp_param


def speak(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param.

    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms)

    Returns
    =======
    (signal, sampling rate) : np.array, int
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``

    """
    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(
        ctypes.byref(audio_sampling_rate),
        ctypes.byref(number_tube_sections),
        ctypes.byref(number_vocal_tract_parameters),
        ctypes.byref(number_glottis_parameters),
        ctypes.byref(number_audio_samples_per_tract_state),
        ctypes.byref(internal_sampling_rate),
    )

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cp_param.shape[0]
    frame_steps = 110  # 2.5 ms
    # within first parenthesis type definition, second initialisation
    # 2000 samples more in the audio signal for safety
    audio = (ctypes.c_double * int((number_frames - 1) * frame_steps + 2000))()

    # init the arrays
    tract_params = (
        ctypes.c_double * (number_frames * number_vocal_tract_parameters.value)
    )()
    glottis_params = (
        ctypes.c_double * (number_frames * number_glottis_parameters.value)
    )()

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19])
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f"Error in vtlSynthesisReset! Errorcode: {failure}")

    # Call the synthesis function. It may calculate a few seconds.
    failure = VTL.vtlSynthBlock(
        ctypes.byref(tract_params),  # input
        ctypes.byref(glottis_params),  # input
        number_frames,
        frame_steps,
        ctypes.byref(audio),  # output
        0,
    )
    if failure != 0:
        raise ValueError("Error in vtlSynthBlock! Errorcode: %i" % failure)

    return (np.array(audio[:-2000]), 44100)


ARTICULATOR = {
    0: "vocal folds",
    1: "tongue",
    2: "lower incisors",
    3: "lower lip",
    4: "other articulator",
    5: "num articulators",
}


def speak_and_extract_tube_information(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param.

    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms)

    Returns
    =======
    (signal, sampling rate, tube_info) : np.array, int, dict
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``
    """
    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(
        ctypes.byref(audio_sampling_rate),
        ctypes.byref(number_tube_sections),
        ctypes.byref(number_vocal_tract_parameters),
        ctypes.byref(number_glottis_parameters),
        ctypes.byref(number_audio_samples_per_tract_state),
        ctypes.byref(internal_sampling_rate),
    )

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cp_param.shape[0]
    frame_steps = 110  # 2.5 ms
    # within first parenthesis type definition, second initialisation
    audio = [(ctypes.c_double * int(frame_steps))() for _ in range(number_frames - 1)]

    # init the arrays
    tract_params = [
        (ctypes.c_double * (number_vocal_tract_parameters.value))()
        for _ in range(number_frames)
    ]
    glottis_params = [
        (ctypes.c_double * (number_glottis_parameters.value))()
        for _ in range(number_frames)
    ]

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19])
    for i in range(number_frames):
        tract_params[i][:] = tmp[i]
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    for i in range(number_frames):
        glottis_params[i][:] = tmp[i]
    del tmp

    # tube sections
    tube_length_cm = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_area_cm2 = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_articulator = [(ctypes.c_int * 40)() for _ in range(number_frames)]
    incisor_pos_cm = [ctypes.c_double(0) for _ in range(number_frames)]
    tongue_tip_side_elevation = [ctypes.c_double(0) for _ in range(number_frames)]
    velum_opening_cm2 = [ctypes.c_double(0) for _ in range(number_frames)]

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f"Error in vtlSynthesisReset! Errorcode: {failure}")

    for i in range(number_frames):
        if i == 0:
            failure = VTL.vtlSynthesisAddTract(
                0,
                ctypes.byref(audio[0]),
                ctypes.byref(tract_params[i]),
                ctypes.byref(glottis_params[i]),
            )
        else:
            failure = VTL.vtlSynthesisAddTract(
                frame_steps,
                ctypes.byref(audio[i - 1]),
                ctypes.byref(tract_params[i]),
                ctypes.byref(glottis_params[i]),
            )
        if failure != 0:
            raise ValueError("Error in vtlSynthesisAddTract! Errorcode: %i" % failure)

        # export
        failure = VTL.vtlTractToTube(
            ctypes.byref(tract_params[i]),
            ctypes.byref(tube_length_cm[i]),
            ctypes.byref(tube_area_cm2[i]),
            ctypes.byref(tube_articulator[i]),
            ctypes.byref(incisor_pos_cm[i]),
            ctypes.byref(tongue_tip_side_elevation[i]),
            ctypes.byref(velum_opening_cm2[i]),
        )

        if failure != 0:
            raise ValueError("Error in vtlTractToTube! Errorcode: %i" % failure)

    audio = np.ascontiguousarray(audio)
    audio.shape = ((number_frames - 1) * frame_steps,)

    arti = [
        [ARTICULATOR[sec] for sec in list(tube_articulator_i)]
        for tube_articulator_i in list(tube_articulator)
    ]
    incisor_pos_cm = [x.value for x in incisor_pos_cm]
    tongue_tip_side_elevation = [x.value for x in tongue_tip_side_elevation]
    velum_opening_cm2 = [x.value for x in velum_opening_cm2]

    tube_info = {
        "tube_length_cm": np.array(tube_length_cm),
        "tube_area_cm2": np.array(tube_area_cm2),
        "tube_articulator": np.array(arti),
        "incisor_pos_cm": np.array(incisor_pos_cm),
        "tongue_tip_side_elevation": np.array(tongue_tip_side_elevation),
        "velum_opening_cm2": np.array(velum_opening_cm2),
    }

    return (audio, 44100, tube_info)


def audio_padding(sig, samplerate, winlen=0.010):
    """
    Pads the signal by half a window length on each side with zeros.

    Parameters
    ==========
    sig : np.array
        the audio signal
    samplerate : int
        sampling rate
    winlen : float
        the window size in seconds

    """
    pad = int(np.ceil(samplerate * winlen) / 2)
    z = np.zeros(pad)
    pad_signal = np.concatenate((z, sig, z))
    return pad_signal


def mel_to_sig(mel, mel_min=0.0):
    """
    creates audio from a normlised log mel spectrogram.

    Parameters
    ==========
    mel : np.array
        normalised log mel spectrogram (n_mel, seq_length)
    mel_min : float
        original min value (default: 0.0)

    Returns
    =======
    (sig, sampling_rate) : (np.array, int)

    """
    mel = mel + mel_min
    mel = inv_normalize_mel_librosa(mel)
    mel = np.array(mel.T, order="C")
    mel = librosa.db_to_amplitude(mel, ref=0.15)
    sig = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=44100,
        n_fft=1024,
        hop_length=220,
        win_length=1024,
        power=1.0,
        fmin=10,
        fmax=12000,
    )
    # there are always 110 data points missing compared to the speak function using VTL
    # add 55 zeros to the beginning and the end
    sig = np.concatenate((np.zeros(55), sig, np.zeros(55)))
    return (sig, 44100)


def plot_cp(cp, file_name):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.65, 0.8, 0.3], ylim=(-3, 3))
    ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.3], xticklabels=[], sharex=ax1, sharey=ax1)
    ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.3], sharex=ax1, sharey=ax1)

    for ii in range(10):
        ax1.plot(cp[:, ii], label=f"param{ii:0d}")
    ax1.legend()
    for ii in range(10, 20):
        ax2.plot(cp[:, ii], label=f"param{ii:0d}")
    ax2.legend()
    for ii in range(20, 30):
        ax3.plot(cp[:, ii], label=f"param{ii:0d}")
    ax3.legend()
    fig.savefig(file_name, dpi=300)
    plt.close("all")


def plot_mel(mel, file_name):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(mel.T, aspect="equal", vmin=-5, vmax=20)
    fig.savefig(file_name, dpi=300)
    plt.close("all")


def stereo_to_mono(wave, which="both"):
    """
    Extract a channel from a stereo wave

    Parameters
    ==========
    wave: np.array
        Input wave data.
    which: {"left", "right", "both"} default = "both"
        if `mono`, `which` indicates whether the *left* or the *right* channel
        should be extracted, or whether *both* channels should be averaged.

    Returns
    =======
    data: numpy.array
    """
    if which == "left":
        return wave[:, 0]
    if which == "right":
        return wave[:, 1]
    return (wave[:, 0] + wave[:, 1]) / 2


def pad_same_to_even_seq_length(array):
    if not array.shape[0] % 2 == 0:
        return np.concatenate((array, array[-1:, :]), axis=0)
    else:
        return array


def half_seq_by_average_pooling(seq):
    if len(seq) % 2:
        seq = pad_same_to_even_seq_length(seq)
    half_seq = (seq[::2, :] + seq[1::2, :]) / 2
    return half_seq


def export_svgs(cps, path="svgs/", hop_length=5):
    """
    hop_length == 5 : roughly 80 frames per second
    hop_length == 16 : roughly 25 frames per second
    """
    n_tract_parameter = 19

    for ii in range(cps.shape[0] // hop_length):
        jj = ii * hop_length

        tract_params = (ctypes.c_double * 19)()
        tract_params[:] = cps[jj, :n_tract_parameter]

        file_name = os.path.join(path, f"tract{ii:05d}.svg")
        file_name = ctypes.c_char_p(file_name.encode())

        if not os.path.exists(path):
            os.mkdir(path)

        VTL.vtlExportTractSvg(tract_params, file_name)


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


rmse_loss = RMSELoss(eps=0)


def RMSE(x1, x2):
    if x1 is None or x2 is None:
        return np.nan
    rmse = rmse_loss(torch.from_numpy(x1), torch.from_numpy(x2))
    return rmse.item()


def get_vel_acc_jerk(trajectory, *, lag=1):
    """returns (velocity, acceleration, jerk) tuple"""
    velocity = (trajectory[lag:, :] - trajectory[:-lag, :]) / lag
    acc = (velocity[1:, :] - velocity[:-1, :]) / 1.0
    jerk = (acc[1:, :] - acc[:-1, :]) / 1.0
    return velocity, acc, jerk


def cp_trajacetory_loss(Y_hat, tgts):
    """
    Calculate additive loss using the RMSE of position velocity , acc and jerk
    
    :param Y_hat: 3D torch.Tensor
        model prediction
    :param tgts: 3D torch.Tensor
        target tensor
    :return loss, pos_loss, vel_loss, acc_loss, jerk_loss: torch.Tensors
        summed total loss with all individual losses
    """

    velocity, acc, jerk = get_vel_acc_jerk(tgts)
    velocity2, acc2, jerk2 = get_vel_acc_jerk(tgts, lag=2)
    velocity4, acc4, jerk4 = get_vel_acc_jerk(tgts, lag=4)

    Y_hat_velocity, Y_hat_acceleration, Y_hat_jerk = get_vel_acc_jerk(Y_hat)
    Y_hat_velocity2, Y_hat_acceleration2, Y_hat_jerk2 = get_vel_acc_jerk(Y_hat, lag=2)
    Y_hat_velocity4, Y_hat_acceleration4, Y_hat_jerk4 = get_vel_acc_jerk(Y_hat, lag=4)

    pos_loss = rmse_loss(Y_hat, tgts)
    vel_loss = (
        rmse_loss(Y_hat_velocity, velocity)
        + rmse_loss(Y_hat_velocity2, velocity2)
        + rmse_loss(Y_hat_velocity4, velocity4)
    )
    jerk_loss = (
        rmse_loss(Y_hat_jerk, jerk)
        + rmse_loss(Y_hat_jerk2, jerk2)
        + rmse_loss(Y_hat_jerk4, jerk4)
    )
    acc_loss = (
        rmse_loss(Y_hat_acceleration, acc)
        + rmse_loss(Y_hat_acceleration2, acc2)
        + rmse_loss(Y_hat_acceleration4, acc4)
    )

    loss = pos_loss + vel_loss + acc_loss + jerk_loss
    return loss, pos_loss, vel_loss, acc_loss, jerk_loss


def add_and_pad(xx, max_len, with_onset_dim=False):
    """
    Pad a sequence with last value to maximal length

    Parameters
    ==========
    xx : np.array
        A 2d sequence to be padded (seq_length, feeatures)
    max_len : int
        maximal length to be padded to
    with_onset_dim : bool
        add one features with 1 for the first time step and rest 0 to indicate
        sound onset
        
    Returns
    =======
    pad_seq : torch.Tensor
        2D padded sequence
    """
    seq_length = xx.shape[0]
    if with_onset_dim:
        onset = np.zeros((seq_length, 1))
        onset[0, 0] = 1
        xx = np.concatenate((xx, onset), axis=1)  # shape len X (features +1)
    padding_size = max_len - seq_length
    padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
    xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    return torch.from_numpy(xx)


def pad_batch_online(lens, data_to_pad, device="cpu", with_onset_dim=False):
    """
    pads and batches data into one single padded batch.

    Parameters
    ==========
    lens : 1D torch.Tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    data_to_pad : series
        series containing the data to pad

    Returns
    =======
    padded_data : torch.Tensors
        Tensors containing the padded and stacked to one batch
    """
    max_len = int(max(lens))
    padded_data = torch.stack(
        list(
            data_to_pad.apply(
                lambda x: add_and_pad(x, max_len, with_onset_dim=with_onset_dim)
            )
        )
    ).to(device)

    return padded_data


def cps_to_ema_and_mesh(cps, file_prefix, *, path=""):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)
    file_prefix : str
        the prefix of the files written
    path : str
        path where to put the output files

    Returns
    =======
    None : None
        all output is writen to files.
    """

    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(
        ctypes.byref(audio_sampling_rate),
        ctypes.byref(number_tube_sections),
        ctypes.byref(number_vocal_tract_parameters),
        ctypes.byref(number_glottis_parameters),
        ctypes.byref(number_audio_samples_per_tract_state),
        ctypes.byref(internal_sampling_rate),
    )

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cps.shape[0]

    # init the arrays
    tract_params = (
        ctypes.c_double * (number_frames * number_vocal_tract_parameters.value)
    )()
    glottis_params = (
        ctypes.c_double * (number_frames * number_glottis_parameters.value)
    )()

    # fill in data
    tmp = np.ascontiguousarray(cps[:, 0:19])
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cps[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    number_ema_points = 3
    surf = (ctypes.c_int * number_ema_points)()
    surf[:] = np.array([16, 16, 16])  # 16 = TONGUE

    vert = (ctypes.c_int * number_ema_points)()
    vert[:] = np.array(
        [115, 225, 335]
    )  # Tongue Back (TB) = 115; Tongue Middle (TM) = 225; Tongue Tip (TT) = 335

    if not os.path.exists(path):
        os.mkdir(path)

    failure = VTL.vtlTractSequenceToEmaAndMesh(
        ctypes.byref(tract_params),
        ctypes.byref(glottis_params),
        number_vocal_tract_parameters,
        number_glottis_parameters,
        number_frames,
        number_ema_points,
        ctypes.byref(surf),
        ctypes.byref(vert),
        path.encode(),
        file_prefix.encode(),
    )
    if failure != 0:
        raise ValueError(
            "Error in vtlTractSequenceToEmaAndMesh! Errorcode: %i" % failure
        )


def cps_to_ema(cps):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)

    Returns
    =======
    emas : pd.DataFrame
        returns the 3D ema points for different virtual EMA sensors in a
        pandas.DataFrame
    """
    with tempfile.TemporaryDirectory(prefix="python_articubench_") as path:
        file_name = "pyndl_util_ema_export"
        cps_to_ema_and_mesh(cps, file_prefix=file_name, path=path)
        emas = pd.read_table(os.path.join(path, f"{file_name}-ema.txt"), sep=" ")
    return emas


def mel_to_tensor(mel):
    torch_mel = mel.copy()
    torch_mel.shape = (1,) + torch_mel.shape
    torch_mel = torch.from_numpy(torch_mel).detach().clone()
    torch_mel = torch_mel.to(device=DEVICE)
    return torch_mel


def round_up_to_even(f):
    return math.ceil(f / 2.0) * 2


def rigid_transform_3d(A, B):
    """
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    Source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py

    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def calculate_roll_pitch_yaw(rotation_matrix):
    """
    Illustration of the rotation matrix / sometimes called 'orientation' matrix

    R = [  R11 , R12 , R13,
           R21 , R22 , R23,
           R31 , R32 , R33 ]

    REMARKS:
    1. this implementation is meant to make the mathematics easy to be deciphered
    from the script, not so much on 'optimized' code.
    You can then optimize it to your own style.

    2. I have utilized naval rigid body terminology here whereby;
    2.1 roll -> rotation about x-axis
    2.2 pitch -> rotation about the y-axis
    2.3 yaw -> rotation about the z-axis (this is pointing 'upwards')

    https://stackoverflow.com/a/64336115

    """

    r11, r12, r13 = rotation_matrix[0, :]
    r21, r22, r23 = rotation_matrix[1, :]
    r31, r32, r33 = rotation_matrix[2, :]

    if r31 != 1 and r31 != -1:
        pitch_1 = -1 * asin(r31)
        pitch_2 = pi - pitch_1
        roll_1 = atan2(r32 / cos(pitch_1), r33 / cos(pitch_1))
        roll_2 = atan2(r32 / cos(pitch_2), r33 / cos(pitch_2))
        yaw_1 = atan2(r21 / cos(pitch_1), r11 / cos(pitch_1))
        yaw_2 = atan2(r21 / cos(pitch_2), r11 / cos(pitch_2))

        # IMPORTANT NOTE here, there is more than one solution but we choose
        # the first for this case for simplicity !
        # You can insert your own domain logic here on how to handle both
        # solutions appropriately (see the reference publication link for more
        # info).
        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
        yaw = 0  # anything (we default this to zero)
        if r31 == -1:
            pitch = pi / 2
            roll = yaw + atan2(r12, r13)
        else:
            pitch = -pi / 2
            roll = -1 * yaw + atan2(-1 * r12, -1 * r13)

    # convert from radians to degrees
    roll = roll * 180 / pi
    pitch = pitch * 180 / pi
    yaw = yaw * 180 / pi

    rxyz_deg = [roll, pitch, yaw]

    return rxyz_deg


def get_tube_info_stepwise(
    tube_length, tube_area, steps=[5, 8, 11, 13, 14, 15, 16], calculate="raw"
):
    length = np.cumsum(tube_length, axis=1)
    section_per_time = []
    for t, len_ in enumerate(length):
        section = []
        for i, step in enumerate(steps[:-1]):
            area = tube_area[
                t, np.where(np.logical_and(len_ >= step, len_ <= steps[i + 1]))
            ]
            if calculate == "raw":
                section += [area]
            elif calculate == "mean":
                section += [np.mean(area)]
            elif calculate == "binary":
                section += [bool(np.sum(area <= 0.001))]
            else:
                raise Exception("calculate must be one of ['raw', 'mean', 'binary']")
        section_per_time += [section]
    return section_per_time


def download_pretrained_weights(*, skip_if_exists=True, verbose=True):
    package_path = DIR
    model_weights_path = os.path.join(package_path, "models")
    if os.path.isdir(model_weights_path):
        if skip_if_exists:
            if verbose:
                print(
                    f"pretrained_models exist already. Skip download. Path is {model_weights_path}"
                )
                print(
                    f'Version of pretrained weights is "{get_pretrained_weights_version()}"'
                )
                print("To forcefully download the weights, use: ")
                print("  `util.download_pretrained_weights(skip_if_exists=False)`")
            return
        shutil.rmtree(model_weights_path)
    zip_file_url = (
        "https://nc.mlcloud.uni-tuebingen.de/index.php/s/EFr8682rnYKYiWz/download"
    )
    if verbose:
        print(f"downloading 50 MB of model weights from {zip_file_url}")
        print(f"saving pretrained weights to {model_weights_path}")
    stream = requests.get(zip_file_url, stream=True)
    zip_file = zipfile.ZipFile(io.BytesIO(stream.content))
    zip_file.extractall(package_path)
    if verbose:
        print(f'Version of pretrained weights is "{get_pretrained_weights_version()}"')


def get_pretrained_weights_version():
    """read and return the version of the pretrained weights, <No version file
    found> if no pretrained weights exist"""
    version_path = os.path.join(DIR, "models/version.txt")
    if not os.path.exists(version_path):
        return f"<No version file found at {version_path}>"
    with open(version_path, "rt") as vfile:
        version = vfile.read().strip()
    return version


def scale_emas_to_vtl(ema_data, x_offset=8, y_offset=0.3, z_offset=-0.3):
    """
    Scales EMA data from ja_halt dataset to approximately scale to VTL coordinates.
    
    Parameters
    ==========
    ema_data : np.array
        EMA points with shape (n_frames, 3) or just (3,) for a single point
    
    Returns
    =======
    scaled_data : np.array
        Scaled EMA data in VTL coordinate system
    """
    #TODO: Vorzeichen identifizieren 
    #TODO: (Transformationsmatrix)
    ema_copy = ema_data.copy()
    
    #convert mm to cm
    ema_copy = ema_copy / 10

    #if is single point
    if len(ema_copy) == 1:

        ema_copy[1], ema_copy[2] = ema_copy[2], ema_copy[1]
        
        ema_copy[0] = ema_copy[0] + x_offset
        ema_copy[1] = ema_copy[1] + y_offset
        ema_copy[2] = ema_copy[2] + z_offset
        
    else:
        
        ema_copy[:, [1, 2]] = ema_copy[:, [2, 1]]
        
        ema_copy[:, 0] = ema_copy[:, 0] + x_offset
        ema_copy[:, 1] = ema_copy[:, 1] + y_offset    
        ema_copy[:, 2] = ema_copy[:, 2] + z_offset  
        
    return ema_copy


def interpolate(length: int, array: np.array):
    """Interpolates the given array to the given length using scipy's Pchip Interpolator function.

    Parameters
    ==========
    length : int
        the length we wish the given array to be interpolatet to
    array : np.array
        the numpy array we wish to interpolate to the given length

    Returns
    =======
    array : np.array
        return an interpolatet numpy array of length "length"."""

    return PchipInterpolator(np.linspace(0, 1, len(array)), array)(
        np.linspace(0, 1, length)
    )


def align_ema(model_ema: np.array, reference_ema: np.array) -> tuple:
    """Align EMA sequences by interpolating shorter to longer length for our model and reference EMA.
    Since usually reference EMA are a little shorter."""
    
    if model_ema is None or reference_ema is None:
        return model_ema, reference_ema

    target_len = max(len(model_ema), len(reference_ema))

    if len(model_ema) == len(reference_ema):
        return model_ema, reference_ema
    elif len(model_ema) < target_len:
        return interpolate(target_len, model_ema), reference_ema
    else:
        return model_ema, interpolate(target_len, reference_ema)
