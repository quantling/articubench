"""
articubench - An Articulatory Speech Synthesis Benchmark
========================================================

*articubench* is an articulatory speech synthesis benchmark, which used the
articulatory speech synthesiser VocalTractLab (VTL; vocaltractlab.de) and
allows to compare different control models for the VTL. *articubench* comes in
three flavours *tiny*, *small*, *normal* and compares control models besides
the resynthesis audio quality in terms of RMSE on log mel spectrograms on the
RMSE of semantic embeddings of the produced audio, formant transitions in the
audio, tongue height in /babi/, /babu/, /baba/ utterances and tongue tip
movement in /sie/ articulations. For now the benchmark is based on the German
language.

"""

print("WARNING! The *articubench* package is still in alpha and the package "
      "  does not "
      "contain all the files you need to execute the functions defined here. "
      "We are working on a good way to distribute the pretrained weights and "
      "data without making the package big.")

import os
import sys
import multiprocessing as mp
from pip._vendor import pkg_resources


__author__ = 'Konstantin Sering, Paul Schmidt-Barbo'
__author_email__ = 'konstantin.sering@uni-tuebingen.de'
__version__ = '0.1.0'
__license__ = 'GPLv3+'
__description__ = ('articubench - An Articulatory Speech Synthesis Benchmark')
__classifiers__ = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Information Analysis',
    ]


def sysinfo():
    """
    Prints system the dependency information

    """
    articubench = pkg_resources.working_set.by_key["articubench"]
    dependencies = [r.project_name for r in articubench.requires()]

    header = ("Paule Information\n"
              "=================\n\n")

    general = ("General Information\n"
               "-------------------\n"
               "Python version: {}\n"
               "Paule version: {}\n\n").format(sys.version.split()[0], __version__)

    uname = os.uname()
    osinfo = ("Operating System\n"
              "----------------\n"
              "OS: {s.sysname} {s.machine}\n"
              "Kernel: {s.release}\n"
              "CPU: {cpu_count}\n").format(s=uname, cpu_count=mp.cpu_count())

    if uname.sysname == "Linux":
        _, *lines = os.popen("free -m").readlines()
        for identifier in ("Mem:", "Swap:"):
            memory = [line for line in lines if identifier in line]
            if len(memory) > 0:
                _, total, used, *_ = memory[0].split()
            else:
                total, used = '?', '?'
            osinfo += "{} {}MiB/{}MiB\n".format(identifier, used, total)

    osinfo += "\n"

    deps = ("Dependencies\n"
            "------------\n")

    deps += "\n".join("{pkg.__name__}: {pkg.__version__}".format(pkg=__import__(dep))
                      for dep in dependencies)

    print(header + general + osinfo + deps)
