======
Scores
======

The scoring system provides a comprehensive evaluation framework for articulatory speech synthesis models. Each subscore measures a specific aspect of synthesis quality, with higher scores indicating better performance. All subscores are normalized against a baseline model and typically range between 0 and 100.

Score Overview
=============

The benchmark distinguishes three main categories:

1. **Articulatory Scores** 
   - Measures physical accuracy of articulatory movements
   - Evaluates naturalness of motion through velocity/jerk analysis
   - Compares virtual tongue movement to EMA data
   - Validates against ultrasound measurements

2. **Acoustic Scores** 
   - Evaluates audio quality and fidelity
   - Analyzes loudness envelope characteristics
   - Compares mel spectrogram features

3. **Semantic Scores**
   - Assesses semantic accuracy of synthesis
   - Evaluates embedding space distances
   - Measures word classification performance


Score Calculation
===============

The scoring process follows these steps:

1. Calculate error per token
2. Average errors across dataset
3. Normalize by baseline model's average error
4. Subtract from 1 and multiply by 100

This ensures:

- No error = score of 100
- Baseline model error = score of 0

Total Score Formula
----------------
.. math::

    S_\text{total} = S_\text{articulatory} + S_\text{semantic} + S_\text{acoustic}

1. **Articulatory Scoring**
-------------------------

The articulatory score combines three subscores with equal weighting:

1.1 Tongue Height Score
~~~~~~~~~~~~~~~~~~~~~

.. math::

   S_{\text{tongue_height}} = 100 \cdot \left( 1 - \frac{\text{mean}_\text{token}(RMSE(\text{height}_\text{synthesis}, \text{height}_\text{ultrasound}))}{\text{baseline model}} \right)

Implementation Details:
- Uses ultrasound measurements (if available) as ground truth
- Computes RMSE for each time step
- Averages across token duration
- Normalizes against baseline performance

1.2 EMA Score
~~~~~~~~~~~~

.. math::

   S_{\text{ema}} = 100 \cdot \left( 4 - \frac{\text{mean}_\text{token}(\text{RMSE}_\text{TT,x}, \text{RMSE}_\text{TT,y},\text{RMSE}_\text{TT,z}, \text{RMSE}_\text{TB,x}, \text{RMSE}_\text{TB,y}, \text{RMSE}_\text{TB,z},}{\text{baseline model}} \right)

Components:
- Tongue Tip (TT) in x,y,z
- Tongue Body (TB) in x,y,z


1.3 Velocity/Jerk Score
~~~~~~~~~~~~~~~~~~~~~~

.. math::
   S_\text{vel\_jerk} = 100 \cdot \left(2 - \frac{mean_\text{token}(max(\text{velocity}_\text{synthesis}))}{max(\text{velocity}_\text{GECO})} - \frac{mean_\text{token}(max(\text{jerk}_\text{synthesis}))}{max(\text{jerk}_\text{GECO})}\right)

Features:
- Computes velocity from CP trajectories
- Calculates jerk (acceleration derivative)
- Applies 99.9% quantile for outlier handling

2. **Acoustic Scoring**
---------------------

The acoustic score combines loudness and spectrogram analysis:

2.1 Loudness Score
~~~~~~~~~~~~~~~~

.. math::
  S_\text{loudness} = 100 \cdot \left( 1 - \frac{mean_\text{token}( RMSE(\text{loudness}_\text{synthesis}, \text{loudness}_\text{recording}))}{\text{baseline model}} \right)

Parameters:
- Window size: 1024 samples
- Time step: 220 samples
- Aggregation: Sum of log-mel coefficients
- Normalization: Per-token baseline

2.2 Spectrogram Score
~~~~~~~~~~~~~~~~~~~

.. math::
  S_\text{spectrogram} = 100 \cdot \left( 1 - \frac{mean_\text{token}(RMSE(\text{spectrogram}_\text{synthesis}, \text{spectrogram}_\text{recording}))}{\text{baseline model}} \right)

Configuration:
- Mel banks: 60
- Frequency range: 10-12000 Hz
- Time shift: 110 samples
- FFT window: 1024 samples

3. **Semantic Scoring**
--------------------

The semantic score evaluates semantic meaning preservation:

3.1 Semantic Distance Score
~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  S_\text{sem\_dist} = 100 \cdot \left( 1 - \frac{mean_\text{token}( RMSE(\text{semantic\_vector}_\text{synthesis}, \text{semantic\_vector}_\text{target}))}{\text{baseline model}} \right)

Features:
- Uses 300-dim fasttext embeddings
- Computes Euclidean distance
- Normalizes against baseline


3.2 Semantic Rank Score
~~~~~~~~~~~~~~~~~~~~~

.. math::
  S_\text{sem\_rank} = 100 \cdot \left( 1 - \frac{ mean_\text{token}(rank_\text{target} - 1))}{4311} \right)

Implementation:
- Reference set: 4311 vectors # TODO: propably not true anymore
- Ranking: Least to most distant # TODO: maybe use cosine similarity instead?



