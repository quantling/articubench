"""
Here all control models that should be bench marked should be defined and
registered to `control_models_to_evaluate`.

The control model should return control parameter trajectories (cps) that can
be used by VTL to synthesize audio. The time step of the cps is 110 samples or
rougly 2.5 Milliseconds. Dimensions are (seq_length, 30), where the first 19
values of the second dimension are the tract values and the 11 remaining are
the glottis values.

Inputs of the control model are: n_samples, target_semantic_vector,
target_audio, sampling_rate)

Memory Usage
============
Measure Memory usage with::

    # start ipython and load the control model

    ps aux | grep ipython

Divide the fith entry VmSize/VSZ [KiB] by 1024 to get [MiB].


"""


import ctypes
import os
import sys

import numpy as np
import torch

#from paule import paule, util
import paule
import util

VTL_NEUTRAL_TRACT = np.array([1.0, -4.75, 0.0, -2.0, -0.07, 0.95, 0.0, -0.1, -0.4, -1.46, 3.5, -1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
VTL_NEUTRAL_TRACT.shape = (1, 19)

VTL_NEUTRAL_GLOTTIS = np.array([120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0])
VTL_NEUTRAL_GLOTTIS.shape = (1, 11)

PAULE_MODEL = paule.Paule(device=torch.device('cpu'))

def synth_baseline_schwa(n_samples, *, target_semantic_vector=None, target_audio=None,
        sampling_rate=None):

    cps = np.zeros((n_samples, 30))
    cps[:, :19] = VTL_NEUTRAL_TRACT
    cps[:, 19:] = VTL_NEUTRAL_GLOTTIS

    # ramp in pressure
    fade_in = min(n_samples, 1000)
    cps[:fade_in, 20] *= np.linspace(0, 1, fade_in)

    return cps


def synth_paule_acoustic_semvec(n_samples, *, target_semantic_vector=None,
        target_audio=None, sampling_rate=None):
    if target_semantic_vector is None and target_audio is None:
        raise ValueError("You have to either give target_semantic_vector or "
                "target_audio and sampling_rate or both targets.")
    elif target_semantic_vector is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="acoustic",
                objective="acoustic_semvec",
                n_outer=10, n_inner=25,
                continue_learning=True,
                add_training_data=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=True)
    elif target_audio is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=n_samples,
                target_acoustic=None,
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=10, n_inner=25,
                continue_learning=True,
                add_training_data=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=True)
    else:  # both
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=n_samples,
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=10, n_inner=25,
                continue_learning=True,
                add_training_data=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=True)
    cps = results.planned_cp.copy()
    return util.inv_normalize_cp(cps)


def synth_paule_fast(n_samples, *, target_semantic_vector=None,
        target_audio=None, sampling_rate=None):
    if target_semantic_vector is None and target_audio is None:
        raise ValueError("You have to either give target_semantic_vector or "
                "target_audio and sampling_rate or both targets.")
    elif target_semantic_vector is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="acoustic",
                objective="acoustic_semvec",
                n_outer=1, n_inner=5,
                continue_learning=False,
                add_training_data=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    elif target_audio is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=n_samples,
                target_acoustic=None,
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=1, n_inner=5,
                continue_learning=False,
                add_training_data=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    else:  # both
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=n_samples,
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=1, n_inner=5,
                continue_learning=True,
                add_training_data=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    cps = results.planned_cp.copy()
    return util.inv_normalize_cp(cps)



control_models_to_evaluate = {'baseline': synth_baseline_schwa,
        'paule_fast': synth_paule_fast}
        #'paule_acoustic_semvec': synth_paule_acoustic_semvec}

