================
Usage
================

This guide provides comprehensive instructions for using articubench, from basic usage to advanced customization.

Quickstart
==========

1. **Basic Benchmark Run**
   .. code:: bash

       python tests/test_articubench.py

2. **Customized Benchmark Run**
   .. code:: bash

       python tests/test_articubench.py --model paule_fast --size small --tasks all

   Available options:
   - --model: Model to benchmark
     * choices: ['all', 'baseline', 'segment', 'paule_fast', 'semvec', 'inverse']
   - --size: Dataset size
     * choices: ['tiny', 'small']
   - --tasks: Tasks to evaluate
     * choices: ['all'] (single tasks WIP)

Basic Usage
==========

1. **Minimal Example**
   .. code-block:: python

       from articubench.score import score
       from articubench.control_models import (
           synth_paule_fast,
       )

       # Run benchmark with different models
       results_segment = score(
           synth_paule_fast,
           size='tiny',
           tasks='all',
           subscores='all',
           return_individual_subscores=True
       )


2. **Score Function Parameters**
   - model: Control model function
   - size: Dataset size to use
   - tasks: Tasks to evaluate
   - subscores: Specific subscores to compute
   - return_individual_subscores: Return detailed results

Advanced Usage
=============

1. **Custom Control Models**

   a. **PAULE Integration**
   .. code-block:: python

       from paule import paule

       # Initialize PAULE model
       PAULE_MODEL = paule.Paule(
           pred_model=FORWARD_MODEL,
           inv_model=INVERSE_MODEL,
           embedder=EMBEDDING_MODEL,
           device=DEVICE
       )
       
       def synth_paule(seq_length, *, target_semantic_vector=None, target_audio=None, sampling_rate=None):
           # Generate control parameters
           results = PAULE_MODEL.plan_resynth(
               target_semvec=target_semantic_vector,
               target_seq_length=int(seq_length // 2)
           )
           cps = results.planned_cp.copy()
           
           # Validate output
           assert cps.shape[0] == seq_length
           
           # Inverse normalize control parameters for VTL
           return util.inv_normalize_cp(cps)

   b. **Custom Model Template**
   .. code-block:: python

       def custom_control_model(seq_length, *, target_semantic_vector=None, target_audio=None, sampling_rate=None):
           """
           Custom control model template.
           
           Args:
               seq_length: Length of control parameters
               target_semantic_vector: Optional semantic vector
               target_audio: Optional audio signal
               sampling_rate: Optional sampling rate
               
           Returns:
               numpy.ndarray: Control parameters of shape (seq_length, 30)
           """
           # Implement your model here
           cps = generate_control_parameters(...)
           return cps

2. **Task-Specific Usage**

   a. **Acoustic-only Task**
   .. code-block:: python

       results = score(
           model,
           tasks=['acoustic'],
           subscores=['articulatory', 'acoustic'],
           target_audio=audio_signal,
           sampling_rate=44100
       )

   b. **Semantic-only Task**
   .. code-block:: python

       results = score(
           model,
           tasks=['semantic'],
           subscores=['semantic'],
           target_semantic_vector=semantic_vector
       )

   c. **Semantic-Acoustic Task**
   .. code-block:: python

       results = score(
           model,
           tasks=['semantic-acoustic'],
           subscores=['all'],
           target_semantic_vector=semantic_vector,
           target_audio=audio_signal,
           sampling_rate=44100
       )




