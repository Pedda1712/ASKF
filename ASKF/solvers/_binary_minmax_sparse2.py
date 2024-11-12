# ruff: noqa: I001
"""
Sample code automatically generated on 2024-11-11 09:47:33

by geno from www.geno-project.org

from input

parameters
  vector y
  matrix Q
  scalar C
  scalar oldsum
  scalar p
variables
  vector a
min
  0.5*(a.*y)'*Q*diag((oldsum*(Q'*(a.*y)).^2)/norm2((Q'*(a.*y)).^2))*Q'*(a.*y)*(sum((Q'*(a.*y)).^2)/norm2((Q'*(a.*y)).^2)).^p-sum(a)
st
  a <= C
  a >= 0
  a'*y == 0


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
    def __init__(self, y, Q, C, oldsum, p, np):
        self.np = np
        self.y = y
        self.Q = Q
        self.C = C
        self.oldsum = oldsum
        self.p = p
        assert isinstance(y, self.np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        assert isinstance(Q, self.np.ndarray)
        dim = Q.shape
        assert len(dim) == 2
        self.Q_rows = dim[0]
        self.Q_cols = dim[1]
        if isinstance(C, self.np.ndarray):
            dim = C.shape
            assert dim == (1,)
            self.C = C[0]
        self.C_rows = 1
        self.C_cols = 1
        if isinstance(oldsum, self.np.ndarray):
            dim = oldsum.shape
            assert dim == (1,)
            self.oldsum = oldsum[0]
        self.oldsum_rows = 1
        self.oldsum_cols = 1
        if isinstance(p, self.np.ndarray):
            dim = p.shape
            assert dim == (1,)
            self.p = p[0]
        self.p_rows = 1
        self.p_cols = 1
        self.a_rows = self.Q_rows
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.y_rows == self.Q_rows == self.a_rows

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.a_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.C, inf)] * self.a_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.aInit = self.np.ones((self.a_rows, self.a_cols))
        return self.aInit.reshape(-1)

    def variables(self, _x):
        a = _x
        return a

    def fAndG(self, _x):
        a = self.variables(_x)
        t_0 = a * self.y
        t_1 = (self.Q.T).dot(t_0)
        t_2 = t_1**2
        t_3 = self.np.linalg.norm(t_2)
        t_4 = (t_0).dot(self.Q) ** 2
        t_5 = self.np.linalg.norm(t_4)
        t_6 = (self.Q).dot((t_2 * t_1))
        t_7 = (t_0).dot(t_6)
        t_8 = self.np.sum(t_4)
        t_9 = t_8 / t_5
        t_10 = ((t_9 ** (self.p - 1)) * self.oldsum) * self.p
        t_11 = (0.5 * ((self.np.sum(t_2) / t_3) ** self.p)) * self.oldsum
        t_12 = t_6 * self.y
        t_13 = t_9**self.p
        t_14 = t_13 * self.oldsum
        t_15 = t_7 * t_12
        f_ = ((t_11 * t_7) / t_3) - self.np.sum(a)
        g_0 = (
            (
                (
                    (
                        (
                            (((t_10 / (t_5**2)) * t_7) * ((self.Q).dot(t_1) * self.y))
                            - self.np.ones(self.a_rows)
                        )
                        - (((t_10 / (t_5**4)) * t_8) * t_15)
                    )
                    + ((t_11 / t_3) * t_12)
                )
                + ((t_14 / t_5) * ((self.Q).dot(((t_1 * t_1) * t_1)) * self.y))
            )
            + (
                (((0.5 * t_13) * self.oldsum) / t_5)
                * ((self.Q).dot((t_1 * t_2)) * self.y)
            )
        ) - ((t_14 / (t_5**3)) * t_15)
        g_ = g_0
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a = self.variables(_x)
        f = (a).dot(self.y)
        return f

    def gradientEqConstraint000(self, _x):
        g_ = self.y
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        gv_ = _v * self.y
        return gv_


def solveI(y, Q, C, oldsum, p, np, verbose, max_iter=3000):
    NLP = GenoNLP(y, Q, C, oldsum, p, np)
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
    a = NLP.variables(result.x)
    return result, a


def solve(
    F,
    _K_old,
    beta,
    gamma,
    delta,
    c,
    y,
    eigenvaluesOld,
    eigenvectors,
    oldsum,
    p,
    np,
    verbose,
    max_iter=3000,
):
    result, alphas = solveI(y, eigenvectors, c, oldsum, p, np, verbose, max_iter)
    # eigvals are only implicit parameter (depend in closed form
    # on alphas), calculate explicitly here
    new_eigvals = eigenvectors.T.dot(alphas * y) ** 2
    new_eigvals = new_eigvals / np.linalg.norm(new_eigvals)
    new_eigvals *= oldsum
    return result, alphas, new_eigvals
