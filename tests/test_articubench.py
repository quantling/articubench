#!/usr/bin/env python3

import pickle
import sys
import os
import argparse

from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment, synth_inverse_paule


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DIR = os.path.dirname(__file__)

def parser():
    parser = argparse.ArgumentParser(description='Articubench Parser for testing models')
    parser.add_argument('--model', choices=['all', 'baseline', 'segment', 'paule_fast', 'semvec', 'inverse'],
                      default='all', help='Test type to run')
    parser.add_argument('--size', choices=['tiny', 'small'],
                      default='tiny', help='Dataset size, for testings purposes "tiny" is preferred.')
    parser.add_argument('--tasks', default='all',
                      help='Tasks to run (default: all)') #TODO: add single tasks here as well, still unsure how to perform this magic since I made tasks to be tuples ...
    return parser.parse_args()




def test_baseline(size='tiny', tasks='all', subscores='all'):
    results = score(synth_baseline_schwa, size=size, tasks=tasks, subscores=subscores, return_individual_subscores=True)
    with open(DIR + 'results_dump/minimal_example_results_baseline_tiny.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)

def test_segment(size='tiny', tasks='all', subscores='all'):
    results = score(synth_baseline_segment, size=size, tasks=tasks, subscores=subscores, return_individual_subscores=True)
    with open(DIR + 'results_dump/minimal_example_results_segment_tiny.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)


def test_paule_fast(size='tiny', tasks='all', subscores='all'):
    results = score(synth_paule_fast, size=size, tasks=tasks, subscores=subscores, return_individual_subscores=True)
    with open(DIR + 'results_dump/test_minimal_example_results_fast_tiny.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)
    return results

def test_semvec(size='tiny', tasks='all', subscores='all'):
    pass
    # results = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)
    # with open(DIR + 'results_dump/minimal_example_results_semvec.pkl', 'wb') as pfile:
    #     pickle.dump((results), pfile)

def test_inverse(size='tiny', tasks='all', subscores='all'):
    results = score(synth_inverse_paule, size=size, tasks=tasks, subscores=subscores, return_individual_subscores=True)
    with open(DIR + 'results_dump/test_minimal_example_results_inverse_tiny.pkl', 'wb') as pfile:
        pickle.dump((results), pfile)
    return results


if __name__=='__main__':
    args = parser()
    if args.model in ['all', 'baseline']:
        test_baseline(args.size, args.tasks)
    if args.model in ['all', 'segment']:
        test_segment(args.size, args.tasks)
    if args.model in ['all', 'paule_fast']:
        test_paule_fast(args.size, args.tasks)
    if args.model in ['all', 'semvec']:
        test_semvec(args.size, args.tasks)
    if args.model in ['all', 'inverse']:
        test_inverse(args.size, args.tasks)