ASKF - Adaptive Subspace Kernel Fusion
============================================================

**ASKF** is a multiple kernel learning (MKL) method that combines and reweights kernel subspace information [1]. This repository contains an effort to provide a sklearn-compatible implementation of the method, which took place as a part of my bachelor's thesis at [THWS](https://fiw.thws.de/) titled "Adaptive Subspace Learning for Multi-Classification and Regression".

As a result, this implementation also provides various extensions of the baseline approach in [1], namely reduced computational complexity, better explainability (minmax variations), experimental holistic treatment of multi-classifcation (VectorizedASKFClassifier) and support for regression (ASKFEstimator), which were introduced by said bachelor's thesis.

This project uses the  [genosolver](https://www.geno-project.org/) [2][3].

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
genosolver can also be installed from the python package index, but installation from source is recommended for GPU support. GPU support further requires an NVidia Cuda installation and the cupy package to be manually installed.

Getting Started
===============
The interface is mostly self-explaniatory. Have a look at the examples folder to get started with classification.

GPU
===
Using genosolver's [1] [2] inbuilt basic GPU support, better scaling on large(r) datasets can be achieved by setting the `gpu` parameter of the various estimators to `true`. You will need an NVidia GPU, and install the [cupy](https://cupy.dev/) python packages manually into your virtual environment.

References
==========
[1] Maximilian Münch, Manuel Röder, Simon Heilig, Christoph Raab, Frank-Michael Schleif. Static and adaptive subspace information fusion for indefinite heterogeneous proximity data, Neurocomputing, Volume 555, 2023, 126635, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2023.126635.

[2] Soeren Laue, Mark Blacher, and Joachim Giesen. Optimization for Classical Machine Learning Problems on the GPU, AAAI 2022.

[3] Soeren Laue, Matthias Mitterreiter, and Joachim Giesen. GENO -- GENeric Optimization for Classical Machine Learning, NeurIPS 2019.
