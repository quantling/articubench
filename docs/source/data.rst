====
Data
====

Overview
========

Articubench provides multiple datasets designed to support different stages of model development and evaluation. Each dataset serves a specific purpose and includes carefully curated data for articulatory speech synthesis benchmarking.

Dataset Types
============

1. Tiny Dataset
--------------

**Purpose**:
- Quick validation during development
- Basic functionality testing
- Debugging and error checking

**Size and Composition**:
- 4 samples
- Reference real-world data for all scoring metrics


**Usage Guidelines**:
- Development testing
- Interface validation
- Error checking
- Quick prototyping

**Data Format**:
Each row contains:
- file: Audio file identifier
- label: Word/class label
- target_semantic_vector: 300-dim fasttext embedding
- target_sig: Mono audio waveform
- target_sr: Sample rate (44.1kHz)
- len_cp: Control parameter length
- reference_cp: Normalized CPs
- reference_tongue_height: Ultrasound data 
- reference_ema_TT: Tongue tip EMA
- reference_ema_TB: Tongue body EMA

2. Small Dataset
---------------

**Purpose**:
- Meaningful benchmarking
- Statistical evaluation
- Model comparison

**Size and Composition**:
- ~2,000 samples total of KEC corpus
- Half consisting of 50 common "high-usage" words (e.g., "ja", "ich", "so") repeated 20 times
- Half consisting of 1000 rare "low-usage" words (e.g., "Oberreferendarin", "Zahnfleisch", "Arbeitsmarkt")

**Data Format**:
Same structure as tiny dataset, with:
- Complete EMA data for each sample



**Usage Guidelines**:
- Model evaluation
- Performance benchmarking
- Statistical analysis
- Comparative studies



