# ruff: noqa: I001
"""
Sample code automatically generated on 2024-11-06 14:15:07

by geno from www.geno-project.org

from input

parameters
  vector y
  matrix Q
  scalar C
  scalar oldsum
variables
  vector a
min
  0.5*(a.*y)'*Q*diag((oldsum*(Q'*(a.*y)).^2)/norm2((Q'*(a.*y)).^2))*Q'*(a.*y)-sum(a)
st
  a <= C
  a >= 0
  a'*y == 0


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf

# from timeit import default_timer as timer

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
    def __init__(self, y, Q, C, oldsum, np):
        self.np = np
        self.y = y
        self.Q = Q
        self.C = C
        self.oldsum = oldsum
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
        self.a_rows = self.y_rows
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.Q_rows == self.a_rows == self.y_rows

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
        t_3 = 0.5 * self.oldsum
        t_4 = self.np.linalg.norm(t_2)
        t_5 = (self.Q).dot((t_2 * t_1))
        t_6 = self.np.linalg.norm(((t_0).dot(self.Q) ** 2))
        t_7 = (t_0).dot(t_5)
        t_8 = t_5 * self.y
        f_ = ((t_3 * t_7) / t_4) - self.np.sum(a)
        g_0 = (
            (
                (((t_3 / t_4) * t_8) - self.np.ones(self.a_rows))
                + ((self.oldsum / t_6) * ((self.Q).dot(((t_1 * t_1) * t_1)) * self.y))
            )
            + ((t_3 / t_6) * ((self.Q).dot((t_1 * t_2)) * self.y))
        ) - ((self.oldsum / (t_6**3)) * (t_7 * t_8))
        g_ = g_0
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a = self.variables(_x)
        f = (a).dot(self.y)
        return f

    def gradientEqConstraint000(self, _x):
        # a = self.variables(_x)
        g_ = self.y
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        # a = self.variables(_x)
        gv_ = _v * self.y
        return gv_


def solveI(y, Q, C, oldsum, np, verbose, max_iter):
    # start = timer()
    NLP = GenoNLP(y, Q, C, oldsum, np)
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
    # elapsed = timer() - start
    # print('solving took %.3f sec' % elapsed)
    return result, a


# hacky, make this solver accept the same parameters as
# all the others, i.e. also return eigenvalues
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
    np,
    verbose,
    max_iter=3000,
):
    oldsum = np.linalg.norm(eigenvaluesOld)
    result, alphas = solveI(y, eigenvectors, c, oldsum, np, verbose, max_iter)
    # eigvals are only implicit parameter (depend in closed form
    # on alphas), calculate explicitly here
    new_eigvals = eigenvectors.T.dot(alphas * y) ** 2
    new_eigvals = new_eigvals / np.linalg.norm(new_eigvals)
    new_eigvals *= oldsum
    return result, alphas, new_eigvals
