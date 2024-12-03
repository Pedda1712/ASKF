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
from ASKF.solvers import (
    canonical_solve,
    canonical_squared_gamma_solve,
    canonical_faster_solve,
    canonical_squared_gamma_faster_solve,
    vo_canonical_solve,
    binary_minmax_solve,
    vo_squared_gamma_solve,
    binary_minmax_sparse_solve,
    binary_minmax_sparse2_solve,
)
from ASKF.utils import get_spectral_properties


class BinaryASKFClassifier(ClassifierMixin, BaseEstimator):
    """An Adaptive Subspace Kernel Fusion based classifier.
    Implements only binary classification.
    Compatible with GridSearchCV and OneVsRestClassifier.

    Parameters
    ----------
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
    p: float, default=2.0
       if variation="minmax-sparse", this controls the sparsity of the learned
       kernel weights, where 0 imposes no sparsity benefit and larger values
       favor sparser solutions (this should be more stable than "minmax-sparse-pnorm")
       if variation="minmax-sparse-pnorm", this controls the sparsity of the learned
       kernel weights, lower than 2 features exponentially more sparse solutions,
       but might become instable for p<1
    max_iter : int, default=200
        Maximum iterations of the underlying genosolver.
    variation : string, default="default"
        ASKF variation to use, may change what the regularization term looks
        like.
        "minmax" | "default", theory-aligned ASKF, related to EasyMKL
                  (!) ignores gamma, delta, beta, p
                  should be the fastest variation
        "minmax-sparse", like "minmax" but takes p parameter to control
                  sparsity
        "canonical-faster", canonical ASKF with usual gamma regularization rewritten without fro-norm
        "canonical", canonical ASKF
        "squared-gamma", canonical ASKF with squared gamma regularization
        "squared-gamma-faster", like squared-gamma but trace replaced by vector-matrix-vector product

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

    _parameter_constraints = {
        "beta": [float, int],
        "gamma": [float, int],
        "delta": [float, int],
        "c": [float, int],
        "subsample_size": [float, int],
        "p": [float, int],
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
        p=2.0,
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
        self.p = p
        self.gpu = gpu
        self._pairwise = True

    def _more_tags(self):
        return {
            "binary_only": True,
            "poor_score": True,
            "pairwise": True,
            "three_d_array": True,
        }

    def _get_solver(self):
        match self.variation:
            case "squared-gamma":
                return canonical_squared_gamma_solve
            case "canonical":
                return canonical_solve
            case "canonical-faster":
                return canonical_faster_solve
            case "squared-gamma-faster":
                return canonical_squared_gamma_faster_solve
            case "default" | "minmax":
                return binary_minmax_solve
            case "minmax-sparse-pnorm":
                return binary_minmax_sparse_solve
            case "minmax-sparse":
                return binary_minmax_sparse2_solve
            case _:
                raise ValueError("unkown variation")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit an ASKF classifier.

        Note: As ASKF is purely kernel based, vectorial inputs
        would not make sense here. Instead, deviating from other
        sklearn classifiers, you need to input an (np)array of
        similarity matrices for the input (which can be constructed
        with ASKFKernels function from _kernels.py).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples, n_kernels)
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
        # input processing so that the sklearn tests pass
        # (or: making the hippo dance)
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
        self._old_eigenvalues = old_eigenvalues
        eigenvectors = eigenprops["eigenvectors"]

        K_old = eigenvectors @ np.diag(old_eigenvalues) @ eigenvectors.T
        eigenvalues = None

        # GENO solver utilizes the GPU through cupy
        m_np = np
        if self.gpu:
            try:
                import cupy as cp

                m_np = cp
            except Exception as e:
                raise RuntimeError(
                    "[ERROR] While attempting to import cupy for GPU support, error ",
                    e,
                    " was raised.",
                )

        F = m_np.asarray(
            (eigenvectors.T @ eigenvectors) * (eigenvectors.T @ eigenvectors)
        )
        mysolver = self._get_solver()
        oldsum = np.linalg.norm(self._old_eigenvalues)

        # solve for result
        try:
            result, self._alphas, eigenvalues = mysolver(
                F,
                m_np.asarray(K_old),
                self.beta,
                self.gamma,
                self.delta,
                self.c,
                m_np.asarray(y),
                m_np.asarray(old_eigenvalues),
                m_np.asarray(eigenvectors),
                oldsum,
                self.p,
                m_np,
                0,
                self.max_iter,
            )
            self.n_iter_ = result.nit
            if self.gpu:
                self._alphas = m_np.asnumpy(self._alphas)
                eigenvalues = m_np.asnumpy(eigenvalues)
        except Exception as e:
            print("[ERROR] an error occurred during solving: ", e)
            self._alphas = np.ones(y.shape)
            eigenvalues = np.ones(eigenvectors.shape[1])

        self._eigenvalues = eigenvalues

        K_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        K_sum = np.zeros((len(y), len(y)))
        for K in Ks:
            K_sum += K

        self._projMatrix = np.dot(K_new, np.linalg.pinv(K_sum))
        b_values = -y + np.sum(self._alphas * y * K_new, axis=1)
        self._bias = np.median(b_values[np.where(self._alphas > 0)])
        self._y = y

        return self

    def decision_function(self, X):
        """ASKF SVM decision function. This predictor requires similarties to the complete training data.

        Parameters
        ----------
        X      : array-like, shape (n_test, n_train, n_kernels)
            similarities between test data and training data in n_kernels
            different kernels
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The decision function for each data point.
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

        K_test_sum = np.zeros(Ktests[0].shape)

        for K_test_orig in Ktests:
            K_test_sum += K_test_orig

        K_test_proj = np.dot(self._projMatrix, K_test_sum.T)

        y_predict = np.dot(self._alphas * self._y, K_test_proj) - self._bias

        return -y_predict

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
            the predicted class
        """
        # Check if fit had been called
        check_is_fitted(self)

        y_predict = np.where(
            self.decision_function(X) < 0,
            self.classes_[0],
            self.classes_[min(1, len(self.classes_) - 1)],
        )

        return y_predict


def mkVecLabel_(cl, len):
    """Make vector label.

    Ref.: https://eprints.soton.ac.uk/261157/1/vosvm_2.pdf

    Parameters
    ----------
    cl, class index starting at 0
    len, how many classes there are
    """
    vec = np.zeros((len, 1))
    for i in range(0, len):
        if i == cl:
            vec[i] = np.sqrt((len - 1) / (len))
        else:
            vec[i] = (-1) / np.sqrt(len * (len - 1))
    return vec


class VectorizedASKFClassifier(ClassifierMixin, BaseEstimator):
    """An Adaptive Subspace Kernel Fusion based classifier.
    Implements a vector-labeled strategy for holisitic treatment
    of multi-classification.

    This comes at the cost of runtime with the underlying solver,
    which makes this theoretically nice but practically not that
    interesting. Other solving strategies might eventually make
    this feasible.

    Parameters
    ----------
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
        like. There's no minmax
        variation because the underlying solver has issues with the gradient
        computation.
        "canonical-faster" | "default", canonical ASKF with vector labels
        "squared-gamma", canonical ASKF with gamma regularisation term squared


    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from ASKF import VectorizedASKFClassifier, ASKFKernels
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = VectorizedASKFClassifier(gamma=100,beta=10,c=10).fit(ASKFKernels([X@X.T]), y) # more kernels are possible here (single kernel for demonstration, otherwise pointless) # doctest:+SKIP
    >>> clf.predict(ASKFKernels([X@X.T])) # doctest:+SKIP
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2,
       1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2,
       2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1])

    """

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
        return {
            "poor_score": True,
            "pairwise": True,
            "three_d_array": True,
            "multioutput_only": True,
        }

    def _get_solver(self):
        match self.variation:
            case "default" | "canonical-faster":
                return vo_canonical_solve
            case "squared-gamma":
                return vo_squared_gamma_solve
            case _:
                raise ValueError("unkown variation")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit an ASKF classifier.

        Note: As ASKF is purely kernel based, vectorial inputs
        would not make sense here. Instead, deviating from other
        sklearn classifiers, you need to input an (np)array of
        similarity matrices for the input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_samples, n_kernels)
            The array of kernel matrices to consider.
        y : array-like, shape (n_samples,)
            The target class values.

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

        # correct labels are indices into the classes array
        _, y = np.unique(y, return_inverse=True)

        # construction of vector labels
        self.Y_ = None
        self.numLabels_ = np.max(y) + 1
        for cl in np.nditer(y):
            vec = mkVecLabel_(cl, self.numLabels_)
            if self.Y_ is None:
                self.Y_ = vec
            else:
                self.Y_ = np.append(self.Y_, vec, axis=1)

        # askf classification
        eigenprops = get_spectral_properties(Ks, self.subsample_size)
        old_eigenvalues = eigenprops["eigenvalues"]
        eigenvectors = eigenprops["eigenvectors"]

        # GENO solver utilizes the GPU through cupy
        m_np = np
        if self.gpu:
            try:
                import cupy as cp

                m_np = cp
            except Exception as e:
                raise RuntimeError(
                    "[ERROR] While attempting to import cupy for GPU support, error ",
                    e,
                    " was raised.",
                )

        Ky = self.Y_.T @ self.Y_
        eigenvalues = None
        F = (eigenvectors.T @ eigenvectors) * (eigenvectors.T @ eigenvectors)
        my_solver = self._get_solver()
        try:
            result, self._alphas, eigenvalues = my_solver(
                m_np.asarray(F),
                None,
                self.beta,
                self.gamma,
                self.delta,
                self.c,
                m_np.asarray(self.Y_),
                m_np.asarray(Ky),
                m_np.asarray(old_eigenvalues),
                m_np.asarray(eigenvectors),
                m_np,
                0,
                self.max_iter,
            )
            self.n_iter_ = result.nit

            if self.gpu:
                self._alphas = m_np.asnumpy(self._alphas)
                eigenvalues = m_np.asnumpy(eigenvalues)
        except Exception as e:
            print("[ERROR]: an error occurred during solving: ", e)
            self._alphas = np.ones(eigenvectors.shape[0])
            eigenvalues = np.ones(eigenvectors.shape[1])

        K_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        K_sum = np.zeros((len(y), len(y)))
        for K in Ks:
            K_sum += K

        self._projMatrix = np.dot(K_new, np.linalg.pinv(K_sum))
        self._svinds = np.where(self._alphas > 0)[0]
        self._bias = -(self.Y_) + (np.multiply(self._alphas, self.Y_) @ K_new)
        self._bias = np.median(self._bias[:, self._svinds], axis=1)
        self._y = y

        return self

    def predict(self, X):
        """ASKF prediction function. This predictor requires similarties to the complete training data.

        Parameters
        ----------
        X      : array-like, shape (n_kernels, n_test, n_train)
            similarities between test data and training data in n_kernels
            different kernels
        Returns
        -------
        y : ndarray, shape (n_samples,)
            the predicted label
        """
        check_is_fitted(self)

        Ktests = []
        if not scipy.sparse.issparse(X):
            X = np.array(X)
        if X.ndim != 3:
            X = self._validate_data(X)
            Ktests = [X @ self._oldX.T]
        else:
            Ktests = np.transpose(X, (2, 0, 1))

        K_test_sum = np.zeros(Ktests[0].shape)

        for K_test_orig in Ktests:
            K_test_sum += K_test_orig

        K_test_proj = (self._projMatrix @ K_test_sum.T).T
        # vo dot product comparison
        scores = []
        for t in range(0, self.numLabels_):
            yt = mkVecLabel_(t, self.numLabels_)
            ky = yt.T @ self.Y_
            sim = np.repeat(-(yt.T @ self._bias), K_test_proj.shape[0]).reshape(
                -1, 1
            ) + (K_test_proj @ np.multiply(self._alphas, ky).T)
            scores.append(sim)

        scores = np.hstack(scores)
        inds = np.argmax(scores, axis=1)

        return self.classes_[inds]
