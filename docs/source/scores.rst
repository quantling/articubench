======
Scores
======

The scores define different metrics on how the performance of the control model is evaluated. Thereby the scores and subscores are defined in a way that a higher score means a better control model. Each subscore usually lies between 0 and 100. In the domain of articulatory speech synthesis we distinguish three important groups of scores. The first group of subscores measures the quality of the articulatory movements. As there is high variability in human articulatory movements there is no single gold standard movement of the articulators given a specific target acoustics. Nevertheless, the articulators of the VTL should follow some distributional properties of the human articulators and should be comparable to sensor measurements of the human tongue. The articubench benchmarks evaluates the velocity and jerk distribution of the cp-trajectories. Furthermore, it compares the virtual tongue movement to electromagnetic articulography (EMA) data at the tongue tip and the tongue body and compares the virtual tongue height to the human tongue height using ultra sound measurement of the mid sagittal plane. The second group of subscores belongs to the semantic domain. This is implemented by looking at the closeness to a target semantic vector embedding and the classification rank in a single word classification setup. The third group of subscores belongs to the acoustic domain. With these subscores the similarity between synthe- sis and target audio recording are evaluated by comparing a loudness envelope and two log-mel spectrograms. Subscores for the f0 and formant transitions will be added later on. Each group score as well as the total score are calculated as the sum of all subscores. To calculate the scores of a control model, the predicted cp-trajectories serve as input to the VTL. After deriving the corresponding audio and virtual tongue movements, scores are calculated in the following way. First an error on each token is calculated and averaged over all tokens in the dataset. In a next step the average error is normalised by the average error of the baseline model in corresponding subscore. Afterwards, the resulting normalised average error is subtracted from 1 and multiplied by 100. This ensures that having no error results in a subscore of 100 and an error of the size of the baseline model results in a subscore of 0. In the equations this normalization is denoted by the text baseline model in the denominator. Most errors are calculated by computing the root mean squared error (RMSE) between resulting synthesis and the target.

Documentation for the Scoring Functions
=======================================

The scoring functions are defined for three different aspects of data: **Articulatory**, **Acoustic**, and **Semantic** features. 


1. **Articulatory Scoring**
----

The `score_articulatory` function calculates an overall articulatory score by combining the results from three sub-scores:

- **Tongue Height**
- **EMA (Electromagnetic Articulography)**
- **Velocity and Jerk**

1.1 `score_articulatory(data, *, task)`
~~~~
- **Parameters**:
  - `data`: A pandas DataFrame containing the necessary data for scoring.
  - `task`: A string representing the task being evaluated (e.g., one of 'copy-synthesis', 'semantic-only', 'acoustic-only').

- **Returns**:
  - `s_articulatory`: The combined articulatory score.
  - A list of the individual scores: `[s_tongue_height, s_ema, s_vel_jerk]`.



1.2 `score_tongue_height(data, task)`
~~~~

.. math::

   S_{\text{tongue_height}} = 100 \cdot \left( 1 - \frac{\text{mean}_\text{token}(RMSE(\text{height}_\text{synthesis}, \text{height}_\text{ultrasound}))}{\text{baseline model}} \right)


- **Description**:
  Calculates a score based on the difference between the predicted tongue height and the reference tongue height.

- **Returns**:
  - `s_tongue_height`: The score for tongue height.


1.3 `score_ema(data, task)`
~~~~

.. math::

   S_{\text{ema}} = 100 \cdot \left( 4 - \frac{\text{mean}_\text{token}(\text{RMSE}_\text{TT,x}, \text{RMSE}_\text{TT,y},\text{RMSE}_\text{TT,z}, \text{RMSE}_\text{TB,x}, \text{RMSE}_\text{TB,y}, \text{RMSE}_\text{TB,z},}{\text{baseline model}} \right)

.. math::
    
   \text{RMSE}_\text{TT,x} = RMSE(\text{tongue_tip}_\text{synthesis, x}, \text{tongue_tip}_\text{ema, x})

- **Description**:
  Calculates a score based on the EMA (Electromagnetic Articulography) data, specifically for the tongue tip (TT) and tongue body (TB).

- **Returns**:
  - `s_ema`: The score for EMA based on the TT and TB data.



1.4 `score_vel_jerk(data, task)`
~~~~
- **Description**:
  Calculates a score based on the velocity and jerk of the cp-trajectories. The score is computed on a logarithmic scale and considers outliers by using the 99.9% quantile for the calculation.

- **Returns**:
  - `s_vel_jerk`: The combined score for velocity and jerk.



2. **Acoustic Scoring**
----

The `score_acoustic` function evaluates the acoustic properties of the data by combining two sub-scores:

- **Loudness**
- **Spectrogram**

2.1 `score_acoustic(data, *, task)`
~~~~
- **Parameters**:
  - `data`: A pandas DataFrame containing the necessary data for scoring.
  - `task`: A string representing the task being evaluated (e.g., one of 'copy-synthesis', 'semantic-only', 'acoustic-only').

- **Returns**:
  - `s_acoustic`: The combined acoustic score.
  - A list of the individual scores: `[s_loudness, s_spectrogram]`.


2.2 `score_loudness(data, task)`
~~~~
- **Description**:
  Calculates a score based on the difference between the predicted loudness and the target loudness.

- **Returns**:
  - `s_loudness`: The score for loudness.

---

2.3 `score_spectrogram(data, task)`
~~~~
- **Description**:
  Calculates a score based on the difference between the predicted log-mel spectrogram and the target spectrogram.

- **Returns**:
  - `s_spectrogram`: The score for the spectrogram.


3. **Semantic Scoring**
----

The `score_semantic` function evaluates the semantic properties of the data by combining two sub-scores:

- **Semantic Distance**
- **Semantic Rank**

3.1 `score_semantic(data, *, task)`
~~~~

- **Parameters**:
  - `data`: A pandas DataFrame containing the necessary data for scoring.
  - `task`: A string representing the task being evaluated (e.g., one of 'copy-synthesis', 'semantic-only', 'acoustic-only').

- **Returns**:
  - `s_semantic`: The combined semantic score.
  - A list of the individual scores: `[s_sem_dist, s_sem_rank]`.


3.2 `score_sem_dist(data, task)`
~~~~
- **Description**:
  Calculates a score based on the semantic distance between the predicted semantic vector and the target semantic vector.

- **Returns**:
  - `s_sem_dist`: The score for semantic distance.


3.3 `score_sem_rank(data, task)`
~~~~
- **Description**:
  Calculates a score based on the rank of the predicted semantic vector compared to a reference set of vectors.

- **Returns**:
  - `s_sem_rank`: The score for semantic rank.


3.4 `sem_rank(semvec, label)`
~~~~
- **Description**:
  Computes the rank of the correct label’s semantic vector relative to other vectors in the dataset. The rank indicates how well the predicted semantic vector matches the true label.

- **Parameters**:
  - `semvec`: The predicted semantic vector.
  - `label`: The true label associated with the data point.

- **Returns**:
  - `rank_target`: The rank of the target label among all possible labels, where a lower rank indicates a better match.

