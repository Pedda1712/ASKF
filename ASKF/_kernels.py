"""Kernel construction for ASKF methods."""

import numpy as np


def ASKFKernels(Ks):
    """Turn a kernel list into input for e.g. BinaryASKFClassifier.fit()"""
    return np.transpose(np.array(Ks), (1, 2, 0))
