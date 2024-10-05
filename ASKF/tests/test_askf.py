# ruff: noqa: I001
"""This file will just show how to write tests for the template classes."""

# Authors: Peter Preinesberger
# License: BSD 3 clause

from sklearn.datasets import make_blobs
import numpy as np
import pytest
from ASKF import BinaryASKFClassifier, ASKFKernels


@pytest.fixture
def data():
    return make_blobs(centers=2, n_samples=20, n_features=2)


def lin_kernel(X1, X2):
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=1):
    sqdist = np.add(
        np.sum(X1**2, 1).reshape(-1, 1), np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    return np.exp(-gamma * sqdist)


def test_askf_classifier(data):
    """Check the internals and behaviour of `ASKF`."""
    X, y = data
    clf = BinaryASKFClassifier()

    clf.fit(ASKFKernels([lin_kernel(X, X), rbf_kernel(X, X)]), y)
    assert hasattr(clf, "classes_")

    y_pred = clf.predict(ASKFKernels([lin_kernel(X, X), rbf_kernel(X, X)]))
    assert y_pred.shape == (X.shape[0],)
