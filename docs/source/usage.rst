================
Usage
================

Minimal example
======


After installing the model a minimal working example can be found in 'articubench/examples/minimal_example.py'.

.. code-block:: python

    from articubench.score import score
    from articubench.control_models import synth_paule_fast, synth_baseline_schwa, synth_paule_acoustic_semvec, synth_baseline_segment


    results_segment = score(synth_baseline_segment, tasks='all', subscores='all', return_individual_subscores=True)
    results_paule = score(synth_paule_fast, tasks='all', subscores='all', return_individual_subscores=True)
    results_baseline = score(synth_baseline_schwa, tasks='all', subscores='all', return_individual_subscores=True)
    results_paule_full = score(synth_paule_acoustic_semvec, tasks='all', subscores='all', return_individual_subscores=True)

Here we have two parts, first the control model which generates our cp-trajectories and the scoring function which evaluates them.

The control models which come from the PAULE package expect at least a sequence length and depending on the task also target audio or target semantic embeddings.

Loading your own control model
=============================

**Loading PAULE**
~~~~
To use one of the pre-configured control models seen above, we need to have the PAULE package installed, afterwards we simply load the PAULE class using one of PAULE's or our own forward, inverse and embedder models.

Then we use the .plan_resynth() function to generate our CPs, as seen in the below example which is specified for the 'acoustic-semvec' task.

In-detail explanations on PAULE are given the respective documentation, therefore we will skip them here.

.. code-block:: python

    from paule import paule

    PAULE_MODEL = paule.Paule(pred_model = FORWARD_MODEL, inv_model = INVERSE_MODEL, embedder = EMBEDDING_MODEL, device=DEVICE)
    
    def synth_paule(seq_length, *, target_semantic_vector=None):

        results = PAULE_MODEL.plan_resynth(target_semvec=target_semantic_vector,
                target_seq_length=int(seq_length // 2))
        cps = results.planned_cp.copy()
        assert cps.shape[0] == seq_length
        return util.inv_normalize_cp(cps)
~~~~                

This example also showcases two of the pitfalls of the PAULE package, first the need to normalize the CPs before returning them and secondly the produced CPs are half the length of the target audio.
Meaning they have half the time resolution of the target audio.

