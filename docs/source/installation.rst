Installation
============

This guide provides comprehensive instructions for installing articubench and its dependencies. Choose the installation method that best suits your needs.

Quick Installation
-----------------

The simplest way to install articubench is through pip:

.. code:: bash

    pip install --user articubench

This will install the latest stable version from PyPI along with all required dependencies.

Detailed Installation
-------------------

1. **System Requirements**
   - Python 3.11 or higher
   - CUDA-capable GPU (recommended)
   - 8GB RAM minimum
   - 1GB free disk space

2. **Dependencies**
   - Python (>=3.9,<=3.12.7)
   - PyTorch (>=1.7.0)
   - NumPy (>=1.23.1)
   - Pandas (>=1.5.3)
   - librosa (>=0.9.0)
   - matplotlib (>=3.7.1)
   - praatio (>=6.0.0)
   - soundfile (>=0.11)
   - tqdm (>=4.65.0)
   - paule (>=0.3.6)
   - llvmlite (>=0.42.0)
   - setuptools (>=69.1.0)
   - resampy (>=0.4.2)


3. **Installation Methods**

   a. **Using pip (Recommended)**
   .. code:: bash

       # Create and activate virtual environment (recommended)
       python -m venv articubench_env
       source articubench_env/bin/activate  # Linux/Mac
       # or
       .\articubench_env\Scripts\activate  # Windows

       # Install articubench
       pip install articubench

   b. **From Source**
   .. code:: bash

       # Clone the repository
       git clone https://github.com/yourusername/articubench.git
       cd articubench

       # Install in development mode
       pip install -e .

   c. **Using Conda**
   .. code:: bash

       # Create new conda environment
       conda create -n articubench python=3.11
       conda activate articubench

       # Install articubench
       pip install articubench


4. **Verification**
   .. code:: python

       import articubench
       print(articubench.__version__)  # Should print version number
       
   .. code:: bash
       python articubench.tests.test_articubench.py

Troubleshooting
--------------

- CUDA not found: Install CUDA toolkit
- Missing dependencies: Check pyproject.toml
- Segment-based model errors: Create additionaly venv as described in Segment-based model code comments


