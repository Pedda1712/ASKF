# ruff : noqa: I001
"""
Sample code automatically generated on 2024-10-07 14:44:21

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
variables
  vector alphas
  vector eigenvalues
min
  0.5*(alphas.*y)'*eigenvectors*diag(eigenvalues)*eigenvectors'*(alphas.*y)-sum(alphas)+beta*sum(eigenvalues)+gamma*(eigenvaluesOld-eigenvalues)'*F*(eigenvaluesOld-eigenvalues)
st
  alphas >= 0
  alphas <= c
  y'*alphas == 0
  eigenvalues >= 0
  sum(abs(eigenvalues)) <= delta*sum(abs(eigenvaluesOld))


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
    def __init__(self, F, beta, gamma, delta, c, y, eigenvaluesOld, eigenvectors, np):
        self.np = np
        self.F = F
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.c = c
        self.y = y
        self.eigenvaluesOld = eigenvaluesOld
        self.eigenvectors = eigenvectors
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
        self.alphas_rows = self.y_rows
        self.alphas_cols = 1
        self.alphas_size = self.alphas_rows * self.alphas_cols
        self.eigenvalues_rows = self.F_rows
        self.eigenvalues_cols = 1
        self.eigenvalues_size = self.eigenvalues_rows * self.eigenvalues_cols
        # the following dim assertions need to hold for this problem
        assert self.eigenvectors_rows == self.y_rows == self.alphas_rows
        assert (
            self.F_cols
            == self.F_rows
            == self.eigenvectors_cols
            == self.eigenvalues_rows
            == self.eigenvaluesOld_rows
        )
        assert (
            self.eigenvalues_rows
            == self.F_rows
            == self.F_cols
            == self.eigenvectors_cols
            == self.eigenvaluesOld_rows
        )

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.alphas_size
        bounds += [0] * self.eigenvalues_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.alphas_size
        bounds += [inf] * self.eigenvalues_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.alphasInit = self.np.zeros((self.alphas_rows, self.alphas_cols))
        self.eigenvaluesInit = self.np.zeros(
            (self.eigenvalues_rows, self.eigenvalues_cols)
        )
        return self.np.hstack(
            (self.alphasInit.reshape(-1), self.eigenvaluesInit.reshape(-1))
        )

    def variables(self, _x):
        alphas = _x[0 : 0 + self.alphas_size]
        eigenvalues = _x[
            0 + self.alphas_size : 0 + self.alphas_size + self.eigenvalues_size
        ]
        return alphas, eigenvalues

    def fAndG(self, _x):
        alphas, eigenvalues = self.variables(_x)
        t_0 = alphas * self.y
        t_1 = self.eigenvaluesOld - eigenvalues
        t_2 = (self.eigenvectors.T).dot(t_0)
        t_3 = (self.eigenvectors).dot((eigenvalues * t_2))
        t_4 = (self.F).dot(t_1)
        f_ = (
            ((0.5 * (t_0).dot(t_3)) - self.np.sum(alphas))
            + (self.beta * self.np.sum(eigenvalues))
        ) + (self.gamma * (t_1).dot(t_4))
        g_0 = ((0.5 * (t_3 * self.y)) - self.np.ones(self.alphas_rows)) + (
            0.5 * ((self.eigenvectors).dot((t_2 * eigenvalues)) * self.y)
        )
        g_1 = (
            ((0.5 * (t_2 * t_2)) + (self.beta * self.np.ones(self.eigenvalues_rows)))
            - (self.gamma * t_4)
        ) - (self.gamma * (self.F.T).dot(t_1))
        g_ = self.np.hstack((g_0, g_1))
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        f = (self.y).dot(alphas)
        return f

    def gradientEqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        g_0 = self.y
        g_1 = self.np.ones(self.eigenvalues_rows) * 0
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        alphas, eigenvalues = self.variables(_x)
        gv_0 = _v * self.y
        gv_1 = self.np.ones(self.eigenvalues_rows) * 0
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        f = self.np.sum(self.np.abs(eigenvalues)) - (
            self.delta * self.np.sum(self.np.abs(self.eigenvaluesOld))
        )
        return f

    def gradientIneqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        g_0 = self.np.ones(self.alphas_rows) * 0
        g_1 = self.np.sign(eigenvalues)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        alphas, eigenvalues = self.variables(_x)
        gv_0 = self.np.ones(self.alphas_rows) * 0
        gv_1 = _v * self.np.sign(eigenvalues)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_


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
    beta = -beta
    # start = timer()
    NLP = GenoNLP(F, beta, gamma, delta, c, y, eigenvaluesOld, eigenvectors, np)
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
    alphas, eigenvalues = NLP.variables(result.x)
    # elapsed = timer() - start
    return result, alphas, eigenvalues
