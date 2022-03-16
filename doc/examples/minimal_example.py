from articubench.score import score
from articubench.control_models import synth_paule_fast, synth_baseline_schwa

#results = score(synth_paule_fast)
results_baseline = score(synth_baseline_schwa, tasks='all', subscores='all')
results_paule = score(synth_paule_fast, tasks='all', subscores='all')
#results = score(synth_paule_fast, tasks=('copy-synthesis',), subscores=('acoustic',))

