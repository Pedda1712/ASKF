# ruff: noqa: I001
"""This is a module containing the ASKF-SVR regressor."""


# Authors: Peter Preinesberger
# License: BSD 3 clause

import numpy as np
import scipy
from sklearn.base import BaseEstimator, _fit_context, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from ASKF.utils.matrix_decomp import get_spectral_properties
from ASKF.solvers import (
    canonical_svr_solve,
    squard_gamma_svr_solve,
    minmax_svr_solve,
    minmax_svr_sparse_solve,
    minmax_svr_sparse2_solve,
)


class ASKFEstimator(RegressorMixin, BaseEstimator):
    """An Adaptive Subspace Kernel Fusion based estimator.
    Based on support vector regression.


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
    epsilon : float, default=1.0
        epsilon tube width, or the allowed deviation from the
        actual value without incurring loss
    subsample_size: float, default=1.0
        How many eigenvectors of the kernel matrices to consider.
        1.0 considers [n_samples] eigenvectors, values lower than 1 lead
        to lower rank internal kernels. "n_m" keeps all eigenvectors.
    p: float, default=2.0
       controls sparsity of learned kernel weights only works for:
       "minmax-sparse", where 0 does not incur sparsity, and larger values
          penalize non-sparse solutions by factor
            'pow((norm1(weights)/norm2(weights)), p)'
       "minmax-sparse-pnorm", where 2 incurs no sparsity,
          and smaller values (toward 0) favor sparser solutions :
            'p_norm(weights)/norm2(weights)'
          this becomes somewhat unstable for p < 1
    max_iter : int, default=200
        Maximum iterations of the underlying genosolver.
    variation : string, default="default"
        ASKF variation to use, may change what the regularization term looks
        like.
        "minmax" | "default", theory-aligned ASKF, related to EasyMKL
                  (!) ignores gamma, delta, beta
                  should be the fastest variation
        "canonical", based on canonical ASKF
        "squared-gamma", canonical with squared gamma regularization

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y = make_friedman2(random_state=42)
    >>> scl = StandardScaler().fit(X)
    >>> X = scl.transform(X)
    >>> K = X @ X.T
    >>> regr = ASKFEstimator(c=10000, epsilon=0.01)
    >>> regr.fit(ASKFKernels([K]),y) # doctest:+SKIP
    >>> regr.score(ASKFKernels([K]),y) # doctest:+SKIP


    """

    _parameter_constraints = {
        "beta": [float, int],
        "gamma": [float, int],
        "delta": [float, int],
        "c": [float, int],
        "epsilon": [float, int],
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
        epsilon=1.0,
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
        self.epsilon = epsilon
        self.subsample_size = subsample_size
        self.p = p
        self.max_iter = max_iter
        self.variation = variation
        self.gpu = gpu
        self._pairwise = True

    def _more_tags(self):
        return {
            "poor_score": True,
            "pairwise": True,
            "three_d_array": True,
        }

    def _get_solver(self):
        match self.variation:
            case "minmax" | "default":
                return minmax_svr_solve
            case "minmax-sparse-pnorm":
                return minmax_svr_sparse_solve
            case "minmax-sparse":
                return minmax_svr_sparse2_solve
            case "canonical":
                return canonical_svr_solve
            case "squared-gamma":
                return squard_gamma_svr_solve
            case _:
                raise ValueError("unkown variation")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit an ASKF regressor.

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
            if y.dtype not in [
                float,
                int,
                np.float64,
                np.float32,
                np.float16,
                np.int64,
                np.int32,
                np.int16,
                np.int8,
            ]:
                raise ValueError("Unknown label type " + str(y.dtype))
            if np.shape(X)[0] != np.shape(X)[1]:
                raise ValueError("Kernel Matrix has to be square!")
            if np.shape(X)[0] == 1:
                raise ValueError("More than one sample required in ASKFEstimator")
            Ks = [X @ X.T]
        else:
            self.classes_ = np.unique(y)
            Ks = np.transpose(X, (2, 0, 1))
        self._oldX = X

        # askf classification
        eigenprops = get_spectral_properties(Ks, self.subsample_size)
        old_eigenvalues = eigenprops["eigenvalues"]
        self._old_eigenvalues = old_eigenvalues
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

        F = m_np.asarray(
            (eigenvectors.T @ eigenvectors) * (eigenvectors.T @ eigenvectors)
        )
        mysolver = self._get_solver()
        oldsum = np.linalg.norm(self._old_eigenvalues)
        try:
            result, self._a1, self._a2, eigenvalues = mysolver(
                m_np.asarray(F),
                self.beta,
                self.gamma,
                self.delta,
                self.c,
                m_np.asarray(y),
                m_np.asarray(old_eigenvalues),
                m_np.asarray(eigenvectors),
                self.epsilon,
                oldsum,
                self.p,
                m_np,
                0,
                self.max_iter,
            )

            self.n_iter_ = result.nit

            if self.gpu:
                self._a1 = m_np.asnumpy(self._a1)
                self._a2 = m_np.asnumpy(self._a2)
                eigenvalues = m_np.asnumpy(eigenvalues)
        except Exception as e:
            print("[ERROR] : an error occurred during solving : ", e)
            self._a1 = np.ones(y.shape)
            self._a2 = np.zeros(y.shape)
            eigenvalues = np.ones(eigenvectors.shape[1])

        self._eigenvalues = eigenvalues
        self._old_eigenvalues = old_eigenvalues

        self._a = self._a1 - self._a2
        K_new = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # print("eigvals ", eigenvalues, " ", old_eigenvalues)

        K_sum = np.zeros((len(y), len(y)))
        for K in Ks:
            K_sum += K

        self._projMatrix = np.dot(K_new, np.linalg.pinv(K_sum))
        self._bias = np.median(
            (y - K_new.dot(self._a))[np.where((self._a1 + self._a2) > 0)]
        )
        if np.isnan(self._bias):
            self._bias = 0
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
            the predicted value
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
        return K_test_proj.T.dot(self._a) + self._bias
