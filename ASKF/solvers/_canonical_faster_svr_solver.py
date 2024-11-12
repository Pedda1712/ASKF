# ruff: noqa: I001
"""
Sample code automatically generated on 2024-11-09 15:07:56

by geno from www.geno-project.org

from input

parameters
  matrix F symmetric
  scalar beta
  scalar gamma
  scalar delta
  scalar c
  vector y
  vector eigenvaluesOld
  matrix eigenvectors
  scalar epsilon
variables
  vector a1
  vector a2
  vector eigenvalues
min
  0.5*(a1-a2)'*eigenvectors*diag(eigenvalues)*eigenvectors'*(a1-a2)-y'*(a1-a2)+epsilon*sum(a1+a2)+beta*sum(eigenvalues)+gamma*((eigenvaluesOld-eigenvalues)'*F*(eigenvaluesOld-eigenvalues)).^0.5
st
  sum(a1) == sum(a2)
  0 <= a1
  0 <= a2
  a2 <= c
  a1 <= c
  eigenvalues >= 0
  sum(abs(eigenvalues)) <= delta*sum(abs(eigenvaluesOld))


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
    def __init__(
        self, F, beta, gamma, delta, c, y, eigenvaluesOld, eigenvectors, epsilon, np
    ):
        self.np = np
        self.F = F
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.c = c
        self.y = y
        self.eigenvaluesOld = eigenvaluesOld
        self.eigenvectors = eigenvectors
        self.epsilon = epsilon
        assert isinstance(F, self.np.ndarray)
        dim = F.shape
        assert len(dim) == 2
        self.F_rows = dim[0]
        self.F_cols = dim[1]
        if isinstance(beta, self.np.ndarray):
            dim = beta.shape
            assert dim == (1,)
            self.beta = beta[0]
        self.beta_rows = 1
        self.beta_cols = 1
        if isinstance(gamma, self.np.ndarray):
            dim = gamma.shape
            assert dim == (1,)
            self.gamma = gamma[0]
        self.gamma_rows = 1
        self.gamma_cols = 1
        if isinstance(delta, self.np.ndarray):
            dim = delta.shape
            assert dim == (1,)
            self.delta = delta[0]
        self.delta_rows = 1
        self.delta_cols = 1
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
        self.a1_rows = self.y_rows
        self.a1_cols = 1
        self.a1_size = self.a1_rows * self.a1_cols
        self.a2_rows = self.y_rows
        self.a2_cols = 1
        self.a2_size = self.a2_rows * self.a2_cols
        self.eigenvalues_rows = self.F_cols
        self.eigenvalues_cols = 1
        self.eigenvalues_size = self.eigenvalues_rows * self.eigenvalues_cols
        # the following dim assertions need to hold for this problem
        assert self.a2_rows == self.y_rows == self.eigenvectors_rows == self.a1_rows
        assert (
            self.eigenvaluesOld_rows
            == self.F_rows
            == self.F_cols
            == self.eigenvectors_cols
            == self.eigenvalues_rows
        )
        assert (
            self.eigenvaluesOld_rows
            == self.F_cols
            == self.eigenvectors_cols
            == self.F_rows
            == self.eigenvalues_rows
        )

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.a1_size
        bounds += [0] * self.a2_size
        bounds += [0] * self.eigenvalues_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.a1_size
        bounds += [min(self.c, inf)] * self.a2_size
        bounds += [inf] * self.eigenvalues_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.a1Init = self.np.zeros((self.a1_rows, self.a1_cols))
        self.a2Init = self.np.zeros((self.a2_rows, self.a2_cols))
        self.eigenvaluesInit = self.np.zeros(
            (self.eigenvalues_rows, self.eigenvalues_cols)
        )
        return self.np.hstack(
            (
                self.a1Init.reshape(-1),
                self.a2Init.reshape(-1),
                self.eigenvaluesInit.reshape(-1),
            )
        )

    def variables(self, _x):
        # fmt: off
        a1 = _x[0 : 0 + self.a1_size]
        a2 = _x[0 + self.a1_size : 0 + self.a1_size + self.a2_size]
        eigenvalues = _x[
            0 + self.a1_size + self.a2_size : 0
            + self.a1_size
            + self.a2_size
            + self.eigenvalues_size
        ]
        # fmt: on
        return a1, a2, eigenvalues

    def fAndG(self, _x):
        a1, a2, eigenvalues = self.variables(_x)
        t_0 = a1 - a2
        t_1 = self.eigenvaluesOld - eigenvalues
        t_2 = (self.eigenvectors.T).dot(t_0)
        t_3 = (self.eigenvectors).dot((eigenvalues * t_2))
        t_4 = (0.5 * t_3) + (0.5 * (self.eigenvectors).dot((t_2 * eigenvalues)))
        t_5 = self.epsilon * self.np.ones(self.a2_rows)
        t_6 = (self.F).dot(t_1)
        t_7 = (t_1).dot(t_6)
        t_8 = -0.5
        t_9 = (self.F.T).dot(t_1)
        f_ = (
            (
                ((0.5 * (t_0).dot(t_3)) - (self.y).dot(t_0))
                + (self.epsilon * self.np.sum((a1 + a2)))
            )
            + (self.beta * self.np.sum(eigenvalues))
        ) + ((t_7**0.5) * self.gamma)
        g_0 = (t_4 - self.y) + t_5
        g_1 = (self.y - t_4) + t_5
        g_2 = (
            ((0.5 * (t_2 * t_2)) + (self.beta * self.np.ones(self.eigenvalues_rows)))
            - ((0.5 * ((t_7**t_8) * self.gamma)) * t_6)
        ) - ((0.5 * (((t_1).dot(t_9) ** t_8) * self.gamma)) * t_9)
        g_ = self.np.hstack((g_0, g_1, g_2))
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a1, a2, eigenvalues = self.variables(_x)
        f = self.np.sum(a1) - self.np.sum(a2)
        return f

    def gradientEqConstraint000(self, _x):
        a1, a2, eigenvalues = self.variables(_x)
        g_0 = self.np.ones(self.a1_rows)
        g_1 = -self.np.ones(self.a2_rows)
        g_2 = self.np.ones(self.eigenvalues_rows) * 0
        g_ = self.np.hstack((g_0, g_1, g_2))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        a1, a2, eigenvalues = self.variables(_x)
        gv_0 = _v * self.np.ones(self.a1_rows)
        gv_1 = -(_v * self.np.ones(self.a2_rows))
        gv_2 = self.np.ones(self.eigenvalues_rows) * 0
        gv_ = self.np.hstack((gv_0, gv_1, gv_2))
        return gv_

    def functionValueIneqConstraint000(self, _x):
        a1, a2, eigenvalues = self.variables(_x)
        f = self.np.sum(self.np.abs(eigenvalues)) - (
            self.delta * self.np.sum(self.np.abs(self.eigenvaluesOld))
        )
        return f

    def gradientIneqConstraint000(self, _x):
        a1, a2, eigenvalues = self.variables(_x)
        g_0 = self.np.ones(self.a1_rows) * 0
        g_1 = self.np.ones(self.a2_rows) * 0
        g_2 = self.np.sign(eigenvalues)
        g_ = self.np.hstack((g_0, g_1, g_2))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        a1, a2, eigenvalues = self.variables(_x)
        gv_0 = self.np.ones(self.a1_rows) * 0
        gv_1 = self.np.ones(self.a2_rows) * 0
        gv_2 = _v * self.np.sign(eigenvalues)
        gv_ = self.np.hstack((gv_0, gv_1, gv_2))
        return gv_


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
    _oldsum,
    _p,
    np,
    verbose,
    max_iter=3000,
):
    beta = -beta
    NLP = GenoNLP(
        F, beta, gamma, delta, c, y, eigenvaluesOld, eigenvectors, epsilon, np
    )
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
        constraints = (
            {
                "type": "eq",
                "fun": NLP.functionValueEqConstraint000,
                "jacprod": NLP.jacProdEqConstraint000,
            },
            {
                "type": "ineq",
                "fun": NLP.functionValueIneqConstraint000,
                "jacprod": NLP.jacProdIneqConstraint000,
            },
        )
        result = minimize(
            NLP.fAndG, x0, lb=lb, ub=ub, options=options, constraints=constraints, np=np
        )
    else:
        # SciPy: for inequality constraints need to change sign f(x) <= 0 -> f(x) >= 0
        constraints = (
            {
                "type": "eq",
                "fun": NLP.functionValueEqConstraint000,
                "jac": NLP.gradientEqConstraint000,
            },
            {
                "type": "ineq",
                "fun": lambda x: -NLP.functionValueIneqConstraint000(x),
                "jac": lambda x: -NLP.gradientIneqConstraint000(x),
            },
        )
        result = minimize(
            NLP.fAndG,
            x0,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lb, ub)),
            constraints=constraints,
        )

    # assemble solution and map back to original problem
    a1, a2, eigenvalues = NLP.variables(result.x)
    return result, a1, a2, eigenvalues
