=====
Tasks
=====

Overview
========
The articubench benchmark provides three distinct methods for initializing the PAULE control model and generating Control Parameters (CPs).

Task Types
=========

Acoustic-only (Copy-Synthesis)
----------------------------
**Input**: Target audio recording (human speech) and sample rate


Purpose:
  - Test model's ability to mimic human speech acoustics
  - Focus on articulatory and acoustic quality
  - No semantic information provided


Semantic-only
------------
**Input**: Target semantic embedding vector, desired duration


Purpose:
  - Generate speech from meaning alone
  - Test semantic-to-articulation mapping
  - Handle one-to-many relationship (multiple valid pronunciations)
  - Also known as "full generation task"


Semantic-Acoustic
---------------
**Input**: Target audio recording with sample rate AND semantic embedding vector with a desired duration


Purpose:
  - Joint optimization of acoustics and meaning
  - Most complete evaluation scenario
  - Balance multiple constraints


