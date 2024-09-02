#!/usr/bin/env python3

import pickle
import sys
import os
import pandas as pd
# Add the parent directory (the first 'articubench' folder) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment

# results_paule_full = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
# results_paule = score(synth_paule_fast, tasks='all', subscores='all', return_individual_subscores=True)
small_preloaded = pd.read_pickle('/home/andre/small_loaded.pkl')
def test_baseline():
    results = score(synth_baseline_schwa, preloaded_data=small_preloaded, tasks='all', subscores='all', return_individual_subscores=True)
    with open('minimal_example_results_baseline_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)

def test_segment():
    results = score(synth_baseline_segment, preloaded_data=small_preloaded, tasks='all', subscores='all', return_individual_subscores=True)
    with open('minimal_example_results_segment_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)


def test_paule_fast():
    #data = pd.read_pickle('/home/andre/small_loaded.pkl')

    #results = score(synth_paule_fast, preloaded_data=small_preloaded, tasks='all', subscores='all', return_individual_subscores=True)
    results = score(synth_paule_fast, size='small', tasks='all', subscores='all', return_individual_subscores=True)
    with open('minimal_example_results_fast_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)

def test_semvec():
    # tool slow
    pass
    # results = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
    # with open('minimal_example_results_semvec.pkl', 'wb') as pfile:
    #     pickle.dump((results), pfile)

if __name__=="__main__":
    test_paule_fast()
    test_baseline()
    test_segment()
