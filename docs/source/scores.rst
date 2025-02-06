======
Scores
======

The scores define different metrics to evaluate control model performance in articulatory speech synthesis. Higher scores indicate better performance, with each subscore typically ranging between 0 and 100.

Groups of Scores
==============

The benchmark distinguishes three main groups of scores:

1. **Articulatory Scores**
  - Measures quality of articulatory movements
  - Evaluates velocity and jerk distribution
  - Compares virtual tongue movement to EMA data
  - Compares virtual tongue height to ultrasound measurements

2. **Semantic Scores**
  - Evaluates closeness of produced to target semantic vector embedding
  - Assesses classification rank in word classification

3. **Acoustic Scores**
  - Compares synthesis and target audio recording
  - Evaluates loudness envelope and log-mel spectrograms
  - Future: f0 and formant transition scores


Score Calculation
===============

Scores are calculated as follows:

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
----

The `score_articulatory` function calculates an overall articulatory score by combining the results from three sub-scores:

- **Tongue Height**
- **EMA (Electromagnetic Articulography)**
- **Velocity and Jerk**


1.1 `score_tongue_height(data, task)`
~~~~

.. math::

   S_{\text{tongue_height}} = 100 \cdot \left( 1 - \frac{\text{mean}_\text{token}(RMSE(\text{height}_\text{synthesis}, \text{height}_\text{ultrasound}))}{\text{baseline model}} \right)



Calculates a score based on the mean RMSE difference between the predicted tongue height and the reference tongue height.



1.2 `score_ema(data, task)`
~~~~

.. math::

   S_{\text{ema}} = 100 \cdot \left( 4 - \frac{\text{mean}_\text{token}(\text{RMSE}_\text{TT,x}, \text{RMSE}_\text{TT,y},\text{RMSE}_\text{TT,z}, \text{RMSE}_\text{TB,x}, \text{RMSE}_\text{TB,y}, \text{RMSE}_\text{TB,z},}{\text{baseline model}} \right)

.. math::
    
   \text{RMSE}_\text{TT,x} = RMSE(\text{tongue_tip}_\text{synthesis, x}, \text{tongue_tip}_\text{ema, x})


Calculates a score on the EMA (Electromagnetic Articulography) data for the tongue tip (TT) and tongue body (TB) based on the mean RMSE difference between synthesis and reference EMA in x, y, z direction.




1.3 `score_vel_jerk(data, task)`
~~~~

.. math::
   S_\text{vel\_jerk} = 100 \cdot \left(2 - \frac{mean_\text{token}(max(\text{velocity}_\text{synthesis}))}{max(\text{velocity}_\text{GECO})} - \frac{mean_\text{token}(max(\text{jerk}_\text{synthesis}))}{max(\text{jerk}_\text{GECO})}\right)


Calculates a score based on the velocity and jerk of the cp-trajectories. The score is computed on a logarithmic scale and considers outliers by using the 99.9% quantile for the calculation.





2. **Acoustic Scoring**
----

The `score_acoustic` function evaluates the acoustic properties of the data by combining two sub-scores:

- **Loudness**
- **Spectrogram**


2.1 `score_loudness(data, task)`
~~~~

.. math::
  S_\text{loudness} = 100 \cdot \left( 1 - \frac{mean_\text{token}( RMSE(\text{loudness}_\text{synthesis}, \text{loudness}_\text{recording}))}{\text{baseline model}} \right)


Calculates a score based on the difference between the predicted loudness and the target loudness. 
Loudness is calculated every 220 samples over a 1024 sample window by summing all log-mel spectrogram entries for each time slice.


2.2 `score_spectrogram(data, task)`
~~~~

.. math::
  S_\text{spectrogram} = 100 \cdot \left( 1 - \frac{mean_\text{token}(RMSE(\text{spectrogram}_\text{synthesis}, \text{spectrogram}_\text{recording}))}{\text{baseline model}} \right) 

Calculates a score based on the difference between the predicted log-mel spectrogram and the target spectrogram.
We use a Mel spectrogram with 60 banks in the frequency range from 10 to 12000 Hz, a time shift of 110 samples and an aggregation window for the Fourier transform of 1024 samples.

3. **Semantic Scoring**
----

The `score_semantic` function evaluates the semantic properties of the data by combining two sub-scores:

- **Semantic Distance**
- **Semantic Rank**


3.1 `score_sem_dist(data, task)`
~~~~

.. math::
  S_\text{sem\_dist} = 100 \cdot \left( 1 - \frac{mean_\text{token}( RMSE(\text{semantic\_vector}_\text{synthesis}, \text{semantic\_vector}_\text{target}))}{\text{baseline model}} \right)


Calculates a score based on the semantic distance between the predicted semantic vector and the target semantic vector.


3.2 `score_sem_rank(data, task)`
~~~~

.. math::
  S_\text{sem\_rank} = 100 \cdot \left( 1 - \frac{ mean_\text{token}(rank_\text{target} - 1))}{4311} \right)


Calculates a score based on the rank of the predicted semantic vector compared to a set of 4311 reference vectors including the target.
  Ranking them least to most distant based on the euclidean distance between our produced compared and the reference vectors.

