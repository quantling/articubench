#!/usr/bin/env python3

import pickle
import sys
import os

from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment, synth_inverse_paule


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DIR = os.path.dirname(__file__)


def test_baseline():
    results = score(synth_baseline_schwa, size='tiny', tasks='all', subscores='all', return_individual_subscores=True)
    with open(DIR + 'results_dump/minimal_example_results_baseline_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)

def test_segment():
    results = score(synth_baseline_segment, size='tiny', tasks='all', subscores='all', return_individual_subscores=True)
    with open(DIR + 'results_dump/minimal_example_results_segment_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)


def test_paule_fast():
    results = score(synth_paule_fast, size='tiny', tasks='all', subscores='all', return_individual_subscores=True)
    with open(DIR + 'results_dump/test_minimal_example_results_fast_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)
    return results

def test_semvec():
    pass
    # results = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
    # with open(DIR + 'results_dump/minimal_example_results_semvec.pkl', 'wb') as pfile:
    #     pickle.dump((results), pfile)

def test_inverse():
    results = score(synth_inverse_paule, size='tiny', tasks=('copy-synthesis', 'semantic-acoustic'), subscores='all', return_individual_subscores=True)
    with open(DIR + 'results_dump/test_minimal_example_results_fast_small.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)
    return results


if __name__=='__main__':
    test_inverse()
    test_baseline()
    test_segment()
    test_paule_fast()