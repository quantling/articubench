====
Data
====

Overview of the data sets included in articubench.

Rationale
=========
Articubench includes two primary datasets: tiny and small. These datasets are designed to support a range of benchmarking tasks from preliminary checks to statistically significant evaluations. Each dataset serves a specific purpose:

- Tiny Dataset: The tiny dataset is a minimal set designed to ensure that models can correctly process inputs and produce reasonable outputs. It is primarily used for quick validation checks during development.

- Small Dataset: The small dataset provides enough data to conduct meaningful benchmarking with some statistical power. It is used to compare the performance of different models on a controlled, yet manageable, dataset.

Data sets
=========

Tiny
----

The tiny dataset is a minimal subset of the full data intended for quick and basic functionality tests. It includes only four rows, each representing a different data instance. The primary purpose of this dataset is to verify that the provided models can correctly intake data and return reasonable results without errors. The small size allows for rapid iteration and debugging during model development.


Each row in the tiny dataset includes the following columns:

    - file: The name of the audio file associated with the data point.
    - label: The label or class associated with the data point.
    - target_semantic_vector: A vector representing the semantic content of the target audio.
    - target_sig: The waveform of the target audio signal.
    - target_sr: The sample rate of the target audio signal.
    - len_cp: The length of the control parameter (cp), calculated based on the duration of the target signal.
    - reference_cp: The reference control parameters, normalized and truncated to match the length of the target signal.
    - reference_tongue_height: Placeholder for future data related to the tongue height (currently set to None).
    - reference_ema_TT: Placeholder for Electromagnetic Articulography (EMA) data for the tongue tip (TT), currently None.
    - reference_ema_TB: Placeholder for EMA data for the tongue body (TB), currently None.

Small
-----

The small dataset is a more comprehensive subset, consisting of approximately 2,000 rows. This dataset is a mixture of a subset of the GECO word corpus and 1,800 pronunciations of the words "ja" and "halt" with their respective EMA points.


The small dataset includes the same columns as the tiny dataset, but with the following additional details:

    - file: Names of files from the GECO corpus and specific pronunciations of "ja" and "halt."
    - label: Corresponding labels for each file.
    - target_semantic_vector: Vectors representing the semantic content of each target.
    - target_sig: The waveform of each target audio signal.
    - target_sr: Sample rates for each target signal.
    - len_cp: Control parameter lengths adjusted for the target signal's duration.
    - reference_cp: Normalized control parameters, truncated to the signal's length.
    - reference_tongue_height: Placeholder for future tongue height data (currently set to None).
    - reference_ema_TT: EMA data for the TT of "ja / halt" or None .
    - reference_ema_TB: EMA data for the TB of "ja / halt" or None .