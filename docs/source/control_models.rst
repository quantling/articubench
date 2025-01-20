Control models
==============

Articubench has multiple control models that generate CPs for the VocalTractLab. Mainly the "Baseline", "Segment-based" and "PAULE" model are currently implemented, while the "Inverse" model is in production.

Baseline
--------

The Baseline model is a simple model that always generates the same CPs for the VocalTractLab, which produce a simple neutral sounding "Schwa" sound when synthesized. 

This model is used as a reference point for the other models to compare against.

Inputs for all Tasks:
    - Sequence length which is equal to the length of the target CPs (or half our signal length)


Segment-based
-------------

The Segment-based model is based on using the Montreal Forced Aligner (MFA) to generate CPs. Here we first resynthesize the original audio and a text file corresponding to the spoken word.
Then we can use the MFA to map text to phonemes and phonemes with the audio to CPs. Generally generating smooth CPs which are quite a good approximation of the original audio.

Inputs can be:
    - target semantic vector 
    - target audio signal with sample rate

If a target semantic vector is not provided, the model will use the target audio signal to generate CPs. Otherwise it will always produce CPs given the semantic vector.

PAULE
-----

PAULE is a machine learning model which generates CPs based on the given Task. It uses a forward model to map CPs to audio, an inverse model to map audio to CPs and an embedder to map audio to a semantic space.
It is also able to learn and update its model weights given the inputs.

Inputs can be:
    - target semantic vector (for semantic-only task)
    - target audio signal with sample rate (for acoustic-only task)
    - both (for acoustic-semvec task)

There are currently two PAULE implementations available, the "Fast" and the "Acoustic-Semvec" model. The "Fast" model is simply a PAULE model with very short trainigns, while the "Acoustic-Semvec" model goes through multiple full training cycles.

Inverse
-------

The Inverse model is a part of PAULE, generating CPs from audio signals. It is currently in production and not yet available for use.

Inputs:
    - target audio signal with sample rate
