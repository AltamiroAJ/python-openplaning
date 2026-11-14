"""Experimental hull transverse dynamics, with explicit source and SI conventions.

No automatic certification or stability verdict is provided.
"""
from .lewandowski import (
    LewandowskiState, RollForceModel, RollForceDerivatives,
    from_planing_boat,
)
from .units import (
    ft_to_m, knots_to_mps, lbf_to_newton, slug_ft3_to_kg_m3,
    slug_ft2_to_kg_m2, lbf_ft_to_newton_m,
)
from .derivatives import (
    TransverseCoefficientState, HullDerivatives, COEFFICIENT_UNITS,
    evaluate_hull_derivatives, coefficient_state_from_planing_boat,
)

__all__ = [
    'LewandowskiState', 'RollForceModel', 'RollForceDerivatives',
    'from_planing_boat', 'ft_to_m', 'knots_to_mps', 'lbf_to_newton',
    'slug_ft3_to_kg_m3', 'slug_ft2_to_kg_m2', 'lbf_ft_to_newton_m',
    'TransverseCoefficientState', 'HullDerivatives', 'COEFFICIENT_UNITS',
    'evaluate_hull_derivatives', 'coefficient_state_from_planing_boat',
]
from .transverse import (
    RigidInertia, LinearTransverseSystem, linearize_transverse_system,
    assemble_lewandowski_2004, from_resolved_planing_boat,
)

__all__ += [
    'RigidInertia', 'LinearTransverseSystem', 'linearize_transverse_system',
    'assemble_lewandowski_2004', 'from_resolved_planing_boat',
]
