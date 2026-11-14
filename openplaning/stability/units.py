"""Exact or CODATA-standard unit factors, only at source-data boundaries.

The numerical solver always receives SI. Published dimensional regression
constants must be audited independently; do not simply multiply all coefficients.
"""
from __future__ import annotations
import math

FT_M = 0.3048
LBF_N = 4.4482216152605
KNOT_MPS = 1852.0 / 3600.0
SLUG_KG = 14.593902937206363


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite dimensional input")
    return value


def ft_to_m(value):
    return _finite(value) * FT_M


def knots_to_mps(value):
    return _finite(value) * KNOT_MPS


def lbf_to_newton(value):
    return _finite(value) * LBF_N


def lbf_ft_to_newton_m(value):
    return _finite(value) * LBF_N * FT_M


def slug_ft3_to_kg_m3(value):
    return _finite(value) * SLUG_KG / FT_M**3


def slug_ft2_to_kg_m2(value):
    return _finite(value) * SLUG_KG * FT_M**2
