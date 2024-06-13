#!/usr/bin/env python3

import pickle

from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment

# results_paule_full = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
# results_paule = score(synth_paule_fast, tasks='all', subscores='all', return_individual_subscores=True)

def test_baseline():
    results = score(synth_baseline_schwa, tasks='all', subscores='all', return_individual_subscores=True)
    with open('minimal_example_results_baseline.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)

def test_segment():
    results = score(synth_baseline_segment, tasks='all', subscores='all', return_individual_subscores=True)
    with open('minimal_example_results_segment.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)


def test_paule_fast():
    # tool slow
    pass
    # results = score(synth_paule_fast, tasks='all', subscores='all', return_individual_subscores=True)
    # with open('minimal_example_results_fast.pkl', 'wb') as pfile:
    #     pickle.dump((results), pfile)

def test_semvec():
    # tool slow
    pass
    # results = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
    # with open('minimal_example_results_semvec.pkl', 'wb') as pfile:
    #     pickle.dump((results), pfile)
