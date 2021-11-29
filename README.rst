===========
articubench
===========

A benchmark to evaluate articulatory speech synthesis systems. This benchmark uses the VocalTractLab [1]_ as its articulatory speech synthesis simulator.

Types of data
=============
* acoustics (wave form, log-melspectrogramms)
* semantics (fast text 300 dim semantic vector for single words)
* mid sagital tongue movement from ultra sound imaging
* electromagnetic articulatory (EMA) sensors on tongue tip and tongue body

Languages
=========
* German
* English (planned)
* Mandarin (planned)

Variants
========
As running the benchmark is computational itensive there are different versions of this benchmark, which require different amounts of articulatory synthesis.

Tiny
----
The smallest possible benchmark to check that everything works, but with no statistical power.

Small
-----

Normal
------

Corpora
=======
Data used here comes from the following speech corpora:

* GECO (only phonetic transscription; duration and phone)
* KEC (EMA data, acoustics)
* Mozilla Common Voice
* baba-babi-babu speech rate (ultra sound; acoustics)


Links
=====
* [1] https://www.vocaltractlab.de/
