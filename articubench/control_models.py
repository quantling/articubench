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
import shutil
import subprocess
from praatio import textgrid
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
import pandas as pd
import torch
import soundfile as sf

from .paule import paule
from .embedding_models import  MelEmbeddingModel


from . import util

DEVICE= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DIR = os.path.dirname(__file__)
LABEL_VECTORS = pd.read_pickle(os.path.join(DIR, "data/lexical_embedding_vectors.pkl"))
LABEL_VECTORS_NP = np.array(list(LABEL_VECTORS.vector))

VTL_NEUTRAL_TRACT = np.array([1.0, -4.75, 0.0, -2.0, -0.07, 0.95, 0.0, -0.1, -0.4, -1.46, 3.5, -1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
VTL_NEUTRAL_TRACT.shape = (1, 19)

VTL_NEUTRAL_GLOTTIS = np.array([120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0])
VTL_NEUTRAL_GLOTTIS.shape = (1, 11)

PAULE_MODEL = paule.Paule(device=torch.device('cpu'))

embedder = MelEmbeddingModel(num_lstm_layers=2, hidden_size=720, dropout=0.7).double()
embedder.load_state_dict(torch.load(
    os.path.join(DIR, "models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
    map_location=DEVICE))
embedder = embedder.to(DEVICE)
embedder.eval()

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
                target_seq_length=int(n_samples // 2),
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
                target_seq_length=int(n_samples // 2),
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
                target_seq_length=int(n_samples // 2),
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
                target_seq_length=int(n_samples // 2),
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


def synth_baseline_segment(n_samples, *, target_semantic_vector=None, target_audio=None,
        sampling_rate=None):
    mfa_check = subprocess.run(['type', '-P', 'mfa'], capture_output=True, text=True).stdout
    if mfa_check == "":
        raise ModuleNotFoundError("Montreal-Forced-Aligner not found.Please install:\n  mfa ---> https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html\n ansiwrap: https://pypi.org/project/ansiwrap/ \n sox: https://anaconda.org/conda-forge/sox")

    else:
        if not os.path.exists(os.path.join(DIR,"temp_input")):
            os.mkdir(os.path.join(DIR,"temp_input"))

        if target_semantic_vector is None and (target_audio is None or sampling_rate is None):
            raise ValueError("You have to either give target_semantic_vector or "
                             "target_audio and sampling_rate or both targets.")
        elif target_audio is None:
            # get label, phones and mean phone durations (based on CommonVoice)
            try:
                label,phones, phone_durations = LABEL_VECTORS[LABEL_VECTORS.vector.astype(str) == str(target_semantic_vector)][["label","phones" ,"phone_durations"]].iloc[0]
            except IndexError as e:
                raise ValueError("Unknown Semantic Vector.") from None
            if not os.path.exists(os.path.join(DIR, "temp_output")):
                os.mkdir(os.path.join(DIR, "temp_output"))
        else:
            if target_semantic_vector is None:
                # predict label
                target_mel = util.librosa_melspec(target_audio,sampling_rate)
                target_mel = util.normalize_mel_librosa(target_mel)
                
                semantic_vector = embedder(util.mel_to_tensor(target_mel), (torch.tensor(target_mel.shape[0]),))[-1, :].detach().cpu().numpy().copy()
                semantic_vector = np.asarray([semantic_vector])
                dist = euclidean_distances(semantic_vector, LABEL_VECTORS_NP)[0]
                label = LABEL_VECTORS.label.iloc[np.argsort(dist)[0]]
                print("Used Embedder to predict label: ", label)
            else:
                # get label
                try:
                    label = LABEL_VECTORS[LABEL_VECTORS.vector.astype(str) == str(target_semantic_vector)].label.iloc[0]
                except IndexError as e:
                    raise ValueError("Unknown Semantic Vector.") from None

            # store input
            sf.write(os.path.join(DIR,'temp_input/target_audio.wav'), target_audio, sampling_rate)
            with open(os.path.join(DIR,'temp_input/target_audio.txt'), 'w') as f:
                f.write(label)

            # align input
            subprocess.run(['mfa', 'model','download','g2p','german_g2p'])
            subprocess.run(['mfa', 'g2p', 'german_g2p', f'{os.path.join(DIR,"temp_input")}',f'{os.path.join(DIR,"temp_input/target_dict.txt")}'])
            subprocess.run(['mfa', 'model', 'download', 'acoustic', 'german'])
            subprocess.run(['mfa', 'configure', '-t', f'{os.path.join(DIR, "temp_output")}'])
            subprocess.run(['mfa', 'align', f'{os.path.join(DIR,"temp_input")}', f'{os.path.join(DIR,"temp_input/target_dict.txt")}', 'german', f'{os.path.join(DIR,"temp_output")}', '--clean'])

            # extract sampa phones
            tg = textgrid.openTextgrid(f'{os.path.join(DIR,"temp_output/temp_input_pretrained_aligner/pretrained_aligner/textgrids/target_audio.TextGrid")}',False)
            word = tg.tierDict['words'].entryList[0]
            phones = list()
            phone_durations = list()
            for phone in tg.tierDict['phones'].entryList:
                if phone.start >= word.end:
                    break
                if phone.start < word.start:
                    continue

                if phone.label in sampa_convert_dict.keys():
                    phones.append(sampa_convert_dict[phone.label])
                else:
                    phones.append(phone.label)
                phone_durations.append(phone.end - phone.start)

        # write seg file
        rows = []
        for i, phone in enumerate(phones):
            row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
            rows.append(row)
        text = "\n".join(rows)
        seg_file_name = str(f'{os.path.join(DIR,"temp_output/target_audio.seg")}')
        with open(seg_file_name, "w") as text_file:
            text_file.write(text)

        # get tract files and gesture score
        seg_file_name = ctypes.c_char_p(seg_file_name.encode())

        ges_file_name = str(f'{os.path.join(DIR,"temp_output/target_audio.ges")}')
        ges_file_name = ctypes.c_char_p(ges_file_name.encode())

        util.VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)
        tract_file_name = str(f'{os.path.join(DIR,"temp_output/target_audio.txt")}')
        c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

        util.VTL.vtlGesturalScoreToTractSequence(ges_file_name, c_tract_file_name)
        cps = util.read_cp(tract_file_name)

        # remove temp folder
        shutil.rmtree(f'{os.path.join(DIR,"temp_input")}')
        shutil.rmtree(f'{os.path.join(DIR,"temp_output")}')
    return cps

control_models_to_evaluate = {'baseline': synth_baseline_schwa,
        'paule_fast': synth_paule_fast}
        #'paule_acoustic_semvec': synth_paule_acoustic_semvec}

