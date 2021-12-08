articubench
===========

A benchmark to evaluate articulatory speech synthesis systems. This benchmark
uses the VocalTractLab [1] as its articulatory speech synthesis simulator.

.. warning::

   This package is not released yet and will be released in March 2022
   alongside the ESSV 2022 conference.


Types of data
-------------
* wave form (acoustics)
* log-melspectrogramms (acoustics)
* formant transitions (acoustics)
* fasttext 300 dim semantic vector for single words (semantics)
* mid sagital tongue movement contour from ultra sound imaging
* electromagnetic articulatory (EMA) sensors on tongue tip and tongue body

Languages
---------
* German
* English (planned)
* Mandarin (planned)

Variants
--------
As running the benchmark is computational itensive there are different versions
of this benchmark, which require different amounts of articulatory synthesis.


Tiny
^^^^
The smallest possible benchmark to check that everything works, but with no
statistical power.


Small
^^^^^
A small benchmark with some statistical power.


Normal
^^^^^^
The standard benchmark, which might take some time to complete.


Corpora
-------
Data used here comes from the following speech corpora:

* GECO (only phonetic transscription; duration and phone)
* KEC (EMA data, acoustics)
* Mozilla Common Voice
* baba-babi-babu speech rate (ultra sound; acoustics)


Prerequisits
------------

For running the benchmark:

* python >=3.8
* praat

For creating the benchmark:

* mfa (Montreal forced aligner)
* VTL 2.3 (included in this repository)


License
-------
* VTL is GPLv3.0+ license

Links
-----

* [1] https://www.vocaltractlab.de/

