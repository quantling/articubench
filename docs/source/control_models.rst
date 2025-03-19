Control Models
==============

Articubench supports multiple control models for generating Control Parameters (CPs) for the VocalTractLab. Each model has different characteristics, requirements, and performance trade-offs.

Model Requirements
-----------------

All control models must meet the following requirements:

1. Input Interface
   - Accept sequence length parameter
   - Support one or more input modalities:
     * target_semantic_vector (300-dim fasttext vector)
     * target_audio (mono signal)
     * sampling_rate (typically 44.1kHz)

2. Output Format
   - Generate CP trajectories matching VTL requirements:
     * Shape: (seq_length, 30)
     * First 19 values: tract parameters
     * Remaining 11 values: glottis parameters
   - Time step: 110 samples (2.5ms at 44.1kHz)

Available Models
--------------

Baseline
--------

The Baseline model serves as a reference point for evaluation. It generates neutral "Schwa" sounds regardless of input, providing a minimum performance baseline.

Characteristics:
- Simplest implementation
- Fastest execution time
- Minimal memory requirements
- Consistent output quality
- Useful for debugging and validation

Input Requirements:
- seq_length: Length of target CPs

Segment-based
-------------

The Segment-based model uses the Montreal Forced Aligner (MFA) to generate CPs from phoneme sequences. It provides a good balance of accuracy and computational efficiency.

Characteristics:
- Phoneme-based approach
- Smooth CP trajectories
- Good acoustic approximation
- Requires MFA installation

Input Requirements:
- target_semantic_vector (optional)
- target_audio + sampling_rate (if no semantic vector)

PAULE
-----

PAULE (Phonetic Articulatory Universal Language Encoder) is a machine learning model that learns to generate CPs through multiple training iterations.

Architecture:
- Forward model: CPs → Audio
- Inverse model: Audio → CPs
- Embedder: Audio → Semantic space

Variants:
1. Fast Version
   - Minimal training iterations
   - Lower accuracy
   - Suitable for development

2. Acoustic-Semvec Version
   - Full training cycles
   - Higher accuracy
   - Recommended for evaluation

Input Requirements:
- target_semantic_vector (semantic-only task)
- target_audio + sampling_rate (acoustic-only task)
- Both inputs (acoustic-semvec task)

Performance Characteristics:
- High GPU memory requirements
- Accuracy improves with training

Inverse
-------

The Inverse model is a specialized component of PAULE that focuses on direct audio-to-CP mapping.

Input Requirements:
- target_audio + sampling_rate

Performance Characteristics:
- Fast execution time
- Low memory requirements


Performance Considerations
------------------------

1. Memory Requirements
   - Baseline: Minimal memory
   - Segment-based: Low memory
   - PAULE: High GPU memory
   - Inverse: Low memory


2. Execution Time
   - Baseline: Fastest
   - Segment-based: Moderate
   - PAULE: Variable (either fast or slow, depends on training)
   - Inverse: Fast

3. Accuracy Trade-offs
   - Baseline: Lowest accuracy
   - Segment-based: Good balance
   - PAULE: Highest potential accuracy
   - Inverse: Moderate accuracy
