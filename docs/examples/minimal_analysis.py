"""
Read the results of the minimal example and visualise them.

"""

import pickle

with open(f'minimal_example_results.pkl', 'rb') as pfile:
    results_baseline, results_segment, results_paule, results_paule_full = pickle.load(pfile)


print("=" * 60)
print("Baseline")
print("=" * 60)
print(results_baseline[1][['task', 'score_total']])
print(results_baseline[1].iloc[0])

print()
print("=" * 60)
print("Segment")
print("=" * 60)
print(results_segment[1][['task', 'score_total']])
print(results_segment[1].iloc[0])
print(results_segment[1].iloc[1])
print(results_segment[1].iloc[2])

print()
print("=" * 60)
print("PAULE fast")
print("=" * 60)
print(results_paule[1][['task', 'score_total']])
print(results_paule[1].iloc[0])
print(results_paule[1].iloc[1])
print(results_paule[1].iloc[2])

print()
print("=" * 60)
print("PAULE")
print("=" * 60)
print(results_paule_full[1][['task', 'score_total']])
print(results_paule_full[1].iloc[0])
print(results_paule_full[1].iloc[1])
print(results_paule_full[1].iloc[2])


