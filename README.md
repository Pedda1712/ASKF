ASKF - Adaptive Subspace Kernel Fusion
============================================================

**ASKF** is a multiple kernel learning method that combines and reweights
kernel subspace information. This repository contains an effort
to provide a sklearn-compatible implementation of the method.

This project uses the `genosolver <https://www.geno-project.org/>`_ [1][2].

Creating a fresh project for use with ASKF on the command line could be done as:
```bash
mkdir newproject && cd newproject
python3 -m venv venv/
. venv/bin/activate
# install recent version of genosolver from source for GPU support
git clone https://github.com/slaue/genosolver
pip install ./genosolver
# install this repository
git clone https://github.com/Pedda1712/ASKF
pip install ./ASKF
# (optional) run examples
cp ASKF/examples/*.py .
python <...>.py
```
genosolver can also be installed from the python package index, but installation from source is recommended for GPU support.

References
==========
[1] Soeren Laue, Mark Blacher, and Joachim Giesen. Optimization for Classical Machine Learning Problems on the GPU, AAAI 2022.
[2] Soeren Laue, Matthias Mitterreiter, and Joachim Giesen. GENO -- GENeric Optimization for Classical Machine Learning, NeurIPS 2019.
