# -*- coding: utf-8 -*-
"""Python port of subgmres.m -- GMRES with residual subvector monitoring."""

from .solver import (
    SubGMRESResult,
    SubGMRESNotHermitianError,
    SubGMRESNotPositiveDefiniteError,
    subgmres,
)
from .operators import coerce_operator

__all__ = [
    "subgmres",
    "SubGMRESResult",
    "SubGMRESNotHermitianError",
    "SubGMRESNotPositiveDefiniteError",
    "coerce_operator",
]
