# ruff : noqa: I001
"""
Sample code automatically generated on 2024-11-09 18:47:33

by geno from www.geno-project.org

from input

parameters
  scalar c
  vector y
  vector eigenvaluesOld
  matrix eigenvectors
  scalar epsilon
  scalar oldsum
variables
  vector a1
  vector a2
min
  0.5*(a1-a2)'*eigenvectors*diag((oldsum*(eigenvectors'*(a1-a2)).*(eigenvectors'*(a1-a2)))/norm2((eigenvectors'*(a1-a2)).*(eigenvectors'*(a1-a2))))*eigenvectors'*(a1-a2)-y'*(a1-a2)+epsilon*sum(a1+a2)
st
  sum(a1) == sum(a2)
  0 <= a1
  0 <= a2
  a2 <= c
  a1 <= c


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf

try:
    from genosolver import minimize, check_version

    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize

    USE_GENO_SOLVER = False
    WRN = (
        "WARNING: GENO solver not installed. Using SciPy solver instead.\n"
        + "Run:     pip install genosolver"
    )
    print("*" * 63)
    print(WRN)
    print("*" * 63)


class GenoNLP:
    def __init__(self, c, y, eigenvaluesOld, eigenvectors, epsilon, oldsum, np):
        self.np = np
        self.c = c
        self.y = y
        self.eigenvaluesOld = eigenvaluesOld
        self.eigenvectors = eigenvectors
        self.epsilon = epsilon
        self.oldsum = oldsum
        if isinstance(c, self.np.ndarray):
            dim = c.shape
            assert dim == (1,)
            self.c = c[0]
        self.c_rows = 1
        self.c_cols = 1
        assert isinstance(y, self.np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        assert isinstance(eigenvaluesOld, self.np.ndarray)
        dim = eigenvaluesOld.shape
        assert len(dim) == 1
        self.eigenvaluesOld_rows = dim[0]
        self.eigenvaluesOld_cols = 1
        assert isinstance(eigenvectors, self.np.ndarray)
        dim = eigenvectors.shape
        assert len(dim) == 2
        self.eigenvectors_rows = dim[0]
        self.eigenvectors_cols = dim[1]
        if isinstance(epsilon, self.np.ndarray):
            dim = epsilon.shape
            assert dim == (1,)
            self.epsilon = epsilon[0]
        self.epsilon_rows = 1
        self.epsilon_cols = 1
        if isinstance(oldsum, self.np.ndarray):
            dim = oldsum.shape
            assert dim == (1,)
            self.oldsum = oldsum[0]
        self.oldsum_rows = 1
        self.oldsum_cols = 1
        self.a1_rows = self.y_rows
        self.a1_cols = 1
        self.a1_size = self.a1_rows * self.a1_cols
        self.a2_rows = self.y_rows
        self.a2_cols = 1
        self.a2_size = self.a2_rows * self.a2_cols
        # the following dim assertions need to hold for this problem
        assert self.a2_rows == self.y_rows == self.eigenvectors_rows == self.a1_rows

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.a1_size
        bounds += [0] * self.a2_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.a1_size
        bounds += [min(self.c, inf)] * self.a2_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.a1Init = self.np.ones((self.a1_rows, self.a1_cols))
        self.a2Init = self.np.zeros((self.a2_rows, self.a2_cols))
        return self.np.hstack((self.a1Init.reshape(-1), self.a2Init.reshape(-1)))

    def variables(self, _x):
        a1 = _x[0 : 0 + self.a1_size]
        a2 = _x[0 + self.a1_size : 0 + self.a1_size + self.a2_size]
        return a1, a2

    def fAndG(self, _x):
        a1, a2 = self.variables(_x)
        t_0 = a1 - a2
        t_1 = (self.eigenvectors.T).dot(t_0)
        t_2 = t_1 * t_1
        t_3 = 0.5 * self.oldsum
        t_4 = self.np.linalg.norm(t_2)
        if t_4 == 0:
            t_4 = 1e-5
        t_5 = (self.eigenvectors).dot((t_2 * t_1))
        t_6 = (t_0).dot(self.eigenvectors)
        t_7 = self.np.linalg.norm((t_6 * t_6))
        if t_7 == 0:
            t_7 = 1e-5
        t_8 = (self.oldsum / (t_7**3)) * (
            (t_0).dot((self.eigenvectors).dot((t_1 * t_2))) * t_5
        )
        t_9 = (((t_3 / t_4) * t_5) + ((self.oldsum / t_7) * t_5)) + ((t_3 / t_7) * t_5)
        t_10 = self.epsilon * self.np.ones(self.a2_rows)
        f_ = (((t_3 * (t_0).dot(t_5)) / t_4) - (self.y).dot(t_0)) + (
            self.epsilon * self.np.sum((a1 + a2))
        )
        g_0 = ((t_9 - t_8) - self.y) + t_10
        g_1 = ((t_8 - t_9) + self.y) + t_10
        g_ = self.np.hstack((g_0, g_1))
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a1, a2 = self.variables(_x)
        f = self.np.sum(a1) - self.np.sum(a2)
        return f

    def gradientEqConstraint000(self, _x):
        a1, a2 = self.variables(_x)
        g_0 = self.np.ones(self.a1_rows)
        g_1 = -self.np.ones(self.a2_rows)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        a1, a2 = self.variables(_x)
        gv_0 = _v * self.np.ones(self.a1_rows)
        gv_1 = -(_v * self.np.ones(self.a2_rows))
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_


def solveI(
    c, y, eigenvaluesOld, eigenvectors, epsilon, oldsum, np, verbose, max_iter=3000
):
    NLP = GenoNLP(c, y, eigenvaluesOld, eigenvectors, epsilon, oldsum, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {
        "eps_pg": 1e-4,
        "constraint_tol": 1e-4,
        "max_iter": max_iter,
        "m": 10,
        "ls": 0,
        "verbose": 5 if verbose else 0,
    }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version("0.1.0")
        constraints = {
            "type": "eq",
            "fun": NLP.functionValueEqConstraint000,
            "jacprod": NLP.jacProdEqConstraint000,
        }
        result = minimize(
            NLP.fAndG, x0, lb=lb, ub=ub, options=options, constraints=constraints, np=np
        )
    else:
        constraints = {
            "type": "eq",
            "fun": NLP.functionValueEqConstraint000,
            "jac": NLP.gradientEqConstraint000,
        }
        result = minimize(
            NLP.fAndG,
            x0,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            constraints=constraints,
        )

    # assemble solution and map back to original problem
    a1, a2 = NLP.variables(result.x)
    return result, a1, a2


def solve(
    F,
    beta,
    gamma,
    delta,
    c,
    y,
    eigenvaluesOld,
    eigenvectors,
    epsilon,
    oldsum,
    p,
    np,
    verbose,
    max_iter=3000,
):
    result, a1, a2 = solveI(
        c, y, eigenvaluesOld, eigenvectors, epsilon, oldsum, np, verbose, max_iter
    )
    # eigvals are only implicit parameter (depend in closed form
    # on alphas), calculate explicitly here
    new_eigvals = eigenvectors.T.dot(a1 - a2) ** 2
    if np.linalg.norm(new_eigvals) == 0:
        new_eigvals *= 0
    else:
        new_eigvals = new_eigvals / np.linalg.norm(new_eigvals)
        new_eigvals *= oldsum
    return result, a1, a2, new_eigvals
