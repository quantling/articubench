Overview
================

Articubench is a benchmark model to evaluate articulatory speech synthesis systems. This benchmark uses the VocalTractLab (VTL) as its articulatory speech synthesis simulator.
First a model like PAULE is used to generate Control Point Trajectories (CPs) for the VocalTractLab. Afterwards these CPs get simulated by the VTL to generate audio.
Lastly our audio gets mapped by an embedder onto a semantic embedding space to infer the intended meaning of our synthesized audio compared to our original target meaning.

Originally the benchmark was designed to benchmark the PAULE model on three tasks: Acoustic only (Copy-Synthesis), Semantic only and Semantic-Acoustic. 
But as long as certain requirements are met, the benchmark can be used to evaluate any control model that generates CPs for the VocalTractLab.
How to use the benchmark is described in the `Usage` section.

After CPs are generated, the VTL 'speaks' them which gives us a signal with a sample rate of 44100. Furthermore we generate tongue height and EMA data from the VTL given our CPs.

Now the Benchmark calculates Scores described in the `Scores` section to evaluate the performance of the control model. 

Since all scores are calculated by comparing to a 'baseline model' which is always performing a "Schwa" sound no matter the input, the baseline model can unexpectedly outperform the control model in some cases.

Specifically the jerk or velocity losses of our CPs and the loudness envelope can be quite good if we are using a very small dataset which contains audio similar to a "schwa".



Implementation Notes
--------------------

- All tasks currently operate on word-level inputs
- cp-trajectories must match VocalTractLab requirements:
    - 30 control parameters per timeframe
    - 2.5ms timeframe resolution (110 samples at 44.1kHz)
- Since CP resolution is higher than our EMA and ultrasound data, the model uses 1d interpolation to match the required resolution
- The lab measured EMA points have been scaled to first go from mm to cm, switched y and z axis and then offset to match the VTL coordinate system.
- EMA points can be visualized using the animation.py 
- Articubench uses multi-processing for all score and data calculations, apart from the intial CP-Generation which is done sequentially on the GPU since the model trains for each CP individually