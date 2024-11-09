# ruff: noqa: I001
# Author Peter Preinesberger
# License: BSD 3 clause

from ._canonical_solver import solve as canonical_solve
from ._canonical_squared_gamma_solver import solve as canonical_squared_gamma_solve
from ._canonical_faster import solve as canonical_faster_solve
from ._canonical_squared_gamma_faster_solver import (
    solve as canonical_squared_gamma_faster_solve,
)
from ._binary_minmax import solve as binary_minmax_solve
from ._canonical_faster_svr_solver import solve as canonical_svr_solve
from ._vo_canonical_solver import solve as vo_canonical_solve
from ._vo_squared_gamma_solver import solve as vo_squared_gamma_solve
from ._squared_gamma_svr_solver import solve as squard_gamma_svr_solve
from ._minmax_svr import solve as minmax_svr_solve

__all__ = [
    "canonical_solve",
    "canonical_squared_gamma_solve",
    "canonical_faster_solve",
    "canonical_squared_gamma_faster_solve",
    "vo_canonical_solve",
    "binary_minmax_solve",
    "vo_squared_gamma_solve",
    "canonical_svr_solve",
    "squard_gamma_svr_solve",
    "minmax_svr_solve",
]
