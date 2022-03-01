articubench
===========

A benchmark to evaluate articulatory speech synthesis systems. This benchmark
uses the VocalTractLab [1] as its articulatory speech synthesis simulator.

.. warning::

   This package is not released yet and will be released in March 2022
   alongside the ESSV 2022 conference.

Installation
------------

::

    pip install articubench


Overview
--------

.. image:: https://raw.githubusercontent.com/quantling/articubench/main/doc/figure/articubench_overview.png
  :width: 800
  :alt: Box and arrow overview of the data flow and tasks of the articubench benchmark.

The benchmarks defines three tasks: semantic only, semantic-acoustic and
acoustic only.


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

* KEC (EMA data, acoustics)
* baba-babi-babu speech rate (ultra sound; acoustics)
* Mozilla Common Voice
* GECO (only phonetic transscription; duration and phone)


Prerequisits
------------

For running the benchmark:

* python >=3.8
* praat
* VTL API 2.5.1quantling (included in this repository)

Additionally, for creating the benchmark:

* mfa (Montreal forced aligner)


License
-------
* VTL is GPLv3.0+ license

Links
-----

* [1] https://www.vocaltractlab.de/

