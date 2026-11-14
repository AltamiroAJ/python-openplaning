"""Experimental transverse dynamics of hard-chine planing hulls.

This subpackage is not a certified stability assessment.
"""
from .lewandowski import (
    LewandowskiState, RollForceModel, RollForceDerivatives,
    from_planing_boat,
)
from .units import (
    ft_to_m, knots_to_mps, lbf_to_newton, slug_ft3_to_kg_m3,
    slug_ft2_to_kg_m2, lbf_ft_to_newton_m,
)

__all__ = [
    "LewandowskiState", "RollForceModel", "RollForceDerivatives",
    "from_planing_boat", "ft_to_m", "knots_to_mps", "lbf_to_newton",
    "slug_ft3_to_kg_m3", "slug_ft2_to_kg_m2", "lbf_ft_to_newton_m",
]
