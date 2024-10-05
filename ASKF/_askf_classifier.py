# ruff: noqa: I001
# ruff: noqa: E501
"""This is a module containing the ASKF-SVM classifier."""

# Authors: Peter Preinesberger
# License: BSD 3 clause

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from ASKF.solvers import canonical_solve
from ASKF.utils import get_spectral_properties


def ASKFKernels(Ks):
    """Turn a kernel list into input for BinaryASKFClassifier.fit()"""
    return np.transpose(np.array(Ks), (1, 2, 0))


class BinaryASKFClassifier(ClassifierMixin, BaseEstimator):
    """An Adaptive Subspace Kernel Fusion based classifier.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    beta : float, default=1.0
        The paramter governing the first regularization term, penalizing low
        subspace weights.
    gamma : float, default=1.0
        The parameter governing the second regulariuation term, penalizing
        large deviations in the kernel matrix.
    delta : float, default=1.0
        Sets an upper limit for the subspace weights.
    c : float, default=1.0
        C parameter of the SVM.
    subsample_size: float, default=1.0
        How many eigenvectors of the kernel matrices to consider.
        1.0 considers [n_samples] eigenvectors, values lower than 1 lead
        to lower rank internal kernels. "n_m" keeps all eigenvectors.
    max_iter : int, default=200
        Maximum iterations of the underlying genosolver.
    variation : string, default="default"
        ASKF variation to use, may change what the regularization term looks
        like.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from ASKF import BinaryASKFClassifier
    >>> X, y = make_blobs(n_samples=50, n_features=2, centers=2, random_state=0)
    >>> clf = BinaryASKFClassifier(beta=1.0, gamma=1.0, delta=1.0, c=1.0).fit(ASKFKernels([X @ X.T]), y) # doctest:+SKIP
    error
    ...
    >>> clf.predict(ASKFKernels([X @ X.T])) # doctest:+SKIP
    array([1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1,
       0, 1, 1, 0, 1, 0])
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "beta": [float, int],
        "gamma": [float, int],
        "delta": [float, int],
        "c": [float, int],
        "subsample_size": [float, int],
        "max_iter": [int],
        "variation": [str],
        "gpu": [bool],
    }

    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        delta=1.0,
        c=1.0,
        subsample_size=1.0,
        max_iter=200,
        variation="default",
        gpu=False,
    ):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.c = c
        self.subsample_size = subsample_size
        self.max_iter = max_iter
        self.variation = variation
        self.gpu = gpu
        self._pairwise = True

    def _more_tags(self):
        return {"binary_only": True, "poor_score": True, "pairwise": True}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, Ks=[]):
        """Fit an ASKF classifier.
        Note: As ASKF is purely kernel based, vectorial inputs
        would not make sense here. Instead, deviating from other
        sklearn classifiers, you need to input an (np)array of
        similarity matrices for the input.

        Parameters
        ----------
        X : array-like, shape (n_kernels, n_samples, n_samples)
            The array of kernel matrices to consider.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
            Can only contain two distinct values.

        Returns
        -------
        self : object
            Returns self.
        """

        Ks = []
        if not scipy.sparse.issparse(X):
            X = np.array(X)
        if X.ndim != 3:
            X, y = self._validate_data(X, y)
            self.classes_ = np.unique(y)
            if len(self.classes_) > 2:
                print("more than 2 classes in BinaryASKFClassifier")
            if len(self.classes_) == 1:
                raise ValueError(
                    "Classifier can't train when only one class is present."
                )
            if np.shape(X)[0] != np.shape(X)[1]:
                raise ValueError("Kernel Matrix has to be square!")
            if np.shape(X)[0] == 1:
                raise ValueError(
                    "More than one sample required in BinaryASKFClassifier"
                )
            Ks = [X @ X.T]
        else:
            self.classes_ = np.unique(y)
            Ks = np.transpose(X, (2, 0, 1))
        self._oldX = X

        check_classification_targets(y)

        # correct labels
        y = np.where(y == self.classes_[0], 1, -1)

        # askf classification
        eigenprops = get_spectral_properties(Ks, self.subsample_size)
        old_eigenvalues = eigenprops["eigenvalues"]
        eigenvectors = eigenprops["eigenvectors"]

        # TODO: for gpu support, replace m_np with cupy object
        m_np = np
        if self.gpu:
            raise RuntimeError("gpu support not implemented yet")

        K_old = eigenvectors @ m_np.diag(old_eigenvalues) @ eigenvectors.T
        eigenvalues = None

        match self.variation:
            case "default" | "canonical" | _:
                result, self._alphas, eigenvalues = canonical_solve(
                    K_old,
                    self.beta,
                    self.gamma,
                    self.delta,
                    self.c,
                    y,
                    old_eigenvalues,
                    eigenvectors,
                    m_np,
                    0,
                    self.max_iter,
                )
                self.n_iter_ = result.nit

        K_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        K_sum = np.zeros((len(y), len(y)))
        for K in Ks:
            K_sum += K

        self._projMatrix = m_np.dot(K_new, m_np.linalg.pinv(K_sum))
        b_values = -y + m_np.sum(self._alphas * y * K_new, axis=1)
        self._bias = m_np.median(b_values[m_np.where(self._alphas > 0)])
        self._y = y

        return self

    def predict(self, X):
        """ASKF prediction function. This predictor requires similarties to the
        complete training data.

        Parameters
        ----------
        X      : array-like, shape (n_kernels, n_test, n_train)
            similarities between test data and training data in n_kernels
            different kernels
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        # Check if fit had been called
        check_is_fitted(self)

        Ktests = []
        if not scipy.sparse.issparse(X):
            X = np.array(X)
        if X.ndim != 3:
            X = self._validate_data(X)
            Ktests = [X @ self._oldX.T]
        else:
            Ktests = np.transpose(X, (2, 0, 1))

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.

        K_test_sum = np.zeros(Ktests[0].shape)

        for K_test_orig in Ktests:
            K_test_sum += K_test_orig

        K_test_proj = np.dot(self._projMatrix, K_test_sum.T)

        y_predict = np.dot(self._alphas * self._y, K_test_proj) - self._bias
        y_predict = np.where(
            y_predict > 0,
            self.classes_[0],
            self.classes_[min(1, len(self.classes_) - 1)],
        )

        return y_predict
