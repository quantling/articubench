"""
Here all control models that should be bench marked should be defined and
registered to `control_models_to_evaluate`.

The control model should return control parameter trajectories (cps) that can
be used by VTL to synthesize audio. The time step of the cps is 110 samples or
rougly 2.5 Milliseconds. Dimensions are (seq_length, 30), where the first 19
values of the second dimension are the tract values and the 11 remaining are
the glottis values.

Inputs of the control model are: seq_length, target_semantic_vector,
target_audio, sampling_rate)

Memory Usage
============
Measure Memory usage with::

    # start ipython and load the control model

    ps aux | grep ipython

Divide the fith entry VmSize/VSZ [KiB] by 1024 to get [MiB]::

    sampa_convert_dict = {
        'etu':'@',
        'atu':'6',
        'al':'a:',
        'el':'e:',
        'il':'i:',
        'ol':'o:',
        'ul':'u:',
        'oel':'2:',
        'uel':'y:',
        'ae':'E:',
        'oe':'9',
        'ue':'Y',
        'ng':'N',
        'eU':'OY'
    }

"""


import ctypes
import math
import os
import pickle
import subprocess
import tempfile

from praatio import textgrid
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from paule import paule
from paule import util as paule_util

from .embedding_models import  MelEmbeddingModel
from . import util


paule_util.download_pretrained_weights()

DEVICE = util.DEVICE
DIR = os.path.dirname(__file__)
LABEL_VECTORS = pd.read_pickle(os.path.join(DIR, "data/lexical_embedding_vectors.pkl"))
LABEL_VECTORS_NP = np.array(list(LABEL_VECTORS.vector))


VTL_NEUTRAL_TRACT = np.array([1.0, -4.75, 0.0, -2.0, -0.07, 0.95, 0.0, -0.1, -0.4, -1.46, 3.5, -1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
VTL_NEUTRAL_TRACT.shape = (1, 19)

VTL_NEUTRAL_GLOTTIS = np.array([120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0])
VTL_NEUTRAL_GLOTTIS.shape = (1, 11)

PAULE_MODEL = paule.Paule(device=DEVICE)

EMBEDDER = MelEmbeddingModel(num_lstm_layers=2, hidden_size=720, dropout=0.7).double()
EMBEDDER.load_state_dict(torch.load(
    os.path.join(DIR, "models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
    map_location=DEVICE,
    weights_only=True))

EMBEDDER = EMBEDDER.to(DEVICE)
EMBEDDER.eval()

INVERSE_MODEL = PAULE_MODEL.inv_model.eval()
INVERSE_GAN = PAULE_MODEL.cp_gen_model.eval()


with open(os.path.join(DIR, "data/sampa_ipa_dict.pkl"), 'rb') as handle:
    sampa_convert_dict = pickle.load(handle)

def synth_baseline_schwa(seq_length, *, target_semantic_vector=None, target_audio=None,
        sampling_rate=None):

    cps = np.zeros((seq_length, 30))
    cps[:, :19] = VTL_NEUTRAL_TRACT
    cps[:, 19:] = VTL_NEUTRAL_GLOTTIS

    # ramp in pressure
    fade_in = min(seq_length, 1000)
    cps[:fade_in, 20] *= np.linspace(0, 1, fade_in)

    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'
    return cps


def synth_paule_acoustic_semvec(seq_length, *, target_semantic_vector=None,
        target_audio=None, sampling_rate=None):
    OUTER = 10
    INNER = 25
    if target_semantic_vector is None and target_audio is None:
        raise ValueError("You have to either give target_semantic_vector or "
                "target_audio and sampling_rate or both targets.")
    elif target_semantic_vector is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="acoustic",
                objective="acoustic_semvec",
                n_outer=OUTER, n_inner=INNER,
                continue_learning=True,
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    elif target_audio is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2),
                target_acoustic=None,
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=OUTER, n_inner=INNER,
                continue_learning=True,
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    else:  # both
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2),
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=OUTER, n_inner=INNER,
                continue_learning=True,
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=1,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    cps = results.planned_cp.copy()
    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'
    return util.inv_normalize_cp(cps)

# I copy pasted this
def synth_paule_not_fast(seq_length, *, target_semantic_vector=None,
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
                n_outer=10, n_inner=24,
                continue_learning=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    elif target_audio is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2),
                target_acoustic=None,
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=10, n_inner=24,
                continue_learning=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    else:  # both
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length= int(seq_length // 2),
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=10, n_inner=24,
                continue_learning=True,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    cps = results.planned_cp.copy()
    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'
    return util.inv_normalize_cp(cps)

def synth_paule_fast(seq_length, *, target_semantic_vector=None,
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
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    elif target_audio is None:
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2),
                target_acoustic=None,
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=1, n_inner=5,
                continue_learning=False,
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    else:  # both
        results = PAULE_MODEL.plan_resynth(learning_rate_planning=0.01,
                learning_rate_learning=0.001,
                target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2),
                target_acoustic=(target_audio, sampling_rate),
                initialize_from="semvec",
                objective="acoustic_semvec",
                n_outer=1, n_inner=5,
                continue_learning=True,
                add_training_data_pred=False,
                add_training_data_inv=False,
                log_ii=5,
                log_semantics=False,
                n_batches=3, batch_size=8, n_epochs=10,
                log_gradients=False,
                plot=False, seed=None, verbose=False)
    cps = results.planned_cp.copy()
    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'
    return util.inv_normalize_cp(cps)


def synth_baseline_segment(seq_length, *, target_semantic_vector=None, target_audio=None,
        sampling_rate=None, verbose=False):
    command = 'conda run -n aligner mfa version'
    output = subprocess.run(command.split(), capture_output=True, text=True).stderr
    if "ERROR" in output:
        raise ModuleNotFoundError(
        """
        Montreal-Forced-Aligner not found.

        Please install mfa into a conda environment named 'aligner', e. g. with::

            conda create -n aligner -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
            conda run -n aligner mfa server init
            conda run -n aligner mfa server stop

        Test if the installation was successful with 'conda run -n aligner mfa version'.

        https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

        To remove the environment with the aligner run::

            conda env remove -n aligner

        """)

    # download mfa data if not already downlaoded
    command = 'conda run -n aligner mfa model list g2p'
    output = subprocess.run(command.split(), capture_output=True, text=True).stdout
    if 'german_mfa' not in output:
        print("Dowloading g2p model...")
        command = 'conda run -n aligner mfa model download g2p german_mfa'
        subprocess.run(command.split())
    command = 'conda run -n aligner mfa model list acoustic'
    output = subprocess.run(command.split(), capture_output=True, text=True).stdout
    if 'german_mfa' not in output:
        print("Dowloading acoustic model...")
        command = 'conda run -n aligner mfa model download acoustic german_mfa'
        subprocess.run(command.split())
    del output

    with tempfile.TemporaryDirectory(prefix='python_articubench_segment_model_') as path:
    #if True:
    #    path = DIR

        if verbose:
            print(f"Temporary folder for segment based approach is: {path}")

        if not os.path.exists(os.path.join(path, "temp_input")):
            os.mkdir(os.path.join(path, "temp_input"))

        if target_semantic_vector is None and (target_audio is None or sampling_rate is None):
            raise ValueError("You have to either give target_semantic_vector or "
                             "target_audio and sampling_rate or both targets.")
        elif target_audio is None:
            # get label, phones and mean phone durations (based on CommonVoice)
            try:
                label, phones, phone_durations = LABEL_VECTORS[LABEL_VECTORS.vector.astype(str) == str(target_semantic_vector)][["label","phones" ,"phone_durations"]].iloc[0]
            except IndexError:
                raise ValueError("Unknown Semantic Vector.") from None
            if not os.path.exists(os.path.join(path, "temp_output")):
                os.mkdir(os.path.join(path, "temp_output"))
        else:
            if target_semantic_vector is None:
                # predict label
                target_mel = util.librosa_melspec(target_audio,sampling_rate)
                target_mel = util.normalize_mel_librosa(target_mel)
                
                semantic_vector = EMBEDDER(util.mel_to_tensor(target_mel), (torch.tensor(target_mel.shape[0]),))[-1, :].detach().cpu().numpy().copy()
                semantic_vector = np.asarray([semantic_vector])
                dist = euclidean_distances(semantic_vector, LABEL_VECTORS_NP)[0]
                label = LABEL_VECTORS.label.iloc[np.argsort(dist)[0]]
                print("Used Embedder to predict label: ", label)
            else:
                # get label
                try:
                    label = LABEL_VECTORS[LABEL_VECTORS.vector.astype(str) == str(target_semantic_vector)].label.iloc[0]
                except IndexError:
                    raise ValueError("Unknown Semantic Vector.") from None

            # store input
            sf.write(os.path.join(path,'temp_input/target_audio.wav'), target_audio, sampling_rate)
            with open(os.path.join(path,'temp_input/target_audio.txt'), 'w') as f:
                f.write(label)

            # align input
            command = 'conda run -n aligner mfa server start'.split()
            print(' '.join(command))
            subprocess.run(command, capture_output=~verbose)
            command = ('conda run -n aligner mfa g2p'.split()
                    + [os.path.join(path, "temp_input"), "german_mfa", os.path.join(path, "temp_input/target_dict.txt"), '--clean', '--overwrite'])
            print(' '.join(command))
            subprocess.run(command, capture_output=~verbose)
            command = 'conda run -n aligner mfa configure -t'.split() + [os.path.join(path, "temp_output")]
            print(' '.join(command))
            subprocess.run(command, capture_output=~verbose)
            command = ('conda run -n aligner mfa align'.split()
                    + [os.path.join(path,"temp_input"),
                       os.path.join(path,"temp_input/target_dict.txt"),
                       'german_mfa',
                       os.path.join(path,"temp_output"),
                       '--clean'])
            print(' '.join(command))
            subprocess.run(command, capture_output=~verbose)

            # extract sampa phones
            tg = textgrid.openTextgrid(os.path.join(path, "temp_output/target_audio.TextGrid"), False)
            word = tg.getTier("words").entries[0]
            phones = list()
            phone_durations = list()
            for phone in tg.getTier("phones").entries:
                if phone.start >= word.end:
                    break
                if phone.start < word.start:
                    continue

                try:
                    phones.append(sampa_convert_dict[phone.label])
                except KeyError:
                    raise ValueError("Unknown Phone transcribed.") from None
                phone_durations.append(phone.end - phone.start)

        # write seg file
        rows = []
        for i, phone in enumerate(phones):
            row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
            rows.append(row)
        text = "\n".join(rows)
        seg_file_name = str(os.path.join(path,"temp_output/target_audio.seg"))
        with open(seg_file_name, "w") as text_file:
            text_file.write(text)

        # get tract files and gesture score
        seg_file_name = ctypes.c_char_p(seg_file_name.encode())

        ges_file_name = str(os.path.join(path,"temp_output/target_audio.ges"))
        ges_file_name = ctypes.c_char_p(ges_file_name.encode())

        util.VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)
        tract_file_name = str(os.path.join(path,"temp_output/target_audio.txt"))
        c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

        util.VTL.vtlGesturalScoreToTractSequence(ges_file_name, c_tract_file_name)
        cps = util.read_cp(tract_file_name)

        # remove temp folder
        #shutil.rmtree(os.path.join(path,"temp_input"))
        #shutil.rmtree(os.path.join(path,"temp_output"))
    current_length = cps.shape[0]

    if current_length < seq_length:
        print(f"WARNING: segment based approach produced cps that are to short"
              f" (seg: {current_length}, target: {seq_length}). We pad same to"
              f" target length.")
        padding_size = seq_length - current_length
        padding = np.tile(cps[-1 :], (padding_size, 1))
        cps = np.concatenate((cps, padding), axis=0)
    elif current_length > seq_length:
        print(f"WARNING: segment based approach produced cps that are to long"
              f" (seg: {current_length}, target: {seq_length}). We crop to the"
              f" target length and take the cps from the middle.")
        start = math.floor((cps.shape[0] - seq_length) / 2)
        end = cps.shape[0] - math.ceil((cps.shape[0] - seq_length) / 2)
        cps = cps[start:end, :]
    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'

    return cps


def synth_inverse_paule(seq_length, *, target_semantic_vector=None,
                  target_audio=None, sampling_rate=None):


    if target_audio is None or sampling_rate is None:
        cp_gen_noise = torch.randn(1, 1, 100).to(DEVICE)
        cps = INVERSE_GAN(cp_gen_noise, seq_length, torch.tensor(target_semantic_vector, device=DEVICE).unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    else:
        audio_mel = util.normalize_mel_librosa(util.librosa_melspec(target_audio, sampling_rate))
        audio_mel_tensor = torch.tensor(audio_mel, device=DEVICE).unsqueeze(0)
        cps = INVERSE_MODEL(audio_mel_tensor).squeeze(0).detach().cpu().numpy()

    assert cps.shape[0] == seq_length, f'cps have length: {cps.shape[0]}, while seq length is: {seq_length}'

    return util.inv_normalize_cp(cps)
    
control_models_to_evaluate = {'baseline': synth_baseline_schwa,
        'paule_fast': synth_paule_fast}
        #'paule_acoustic_semvec': synth_paule_acoustic_semvec}

