import pickle

from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment

results_segment = score(synth_baseline_segment, tasks='all', subscores='all', return_individual_subscores=True)
results_paule = score(synth_paule_fast, tasks='all', subscores='all', return_individual_subscores=True)
results_baseline = score(synth_baseline_schwa, tasks='all', subscores='all', return_individual_subscores=True)
results_paule_full = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)


#results = score(synth_paule_fast)
#results = score(synth_paule_fast, tasks=('copy-synthesis',), subscores=('acoustic',))
#results_paule_full = score(synth_paule_acoustic_semvec, tasks=['copy-synthesis',], subscores='all', return_individual_subscores=True)
#results_paule = score(synth_paule_fast, tasks=['copy-synthesis',], subscores='all', return_individual_subscores=True)
#with open('memory_test.pkl', 'wb') as pfile:
#    pickle.dump((results_paule_full), pfile)

with open('minimal_example_results.pkl', 'wb') as pfile:
    pickle.dump((results_baseline, results_segment, results_paule, results_paule_full), pfile)

