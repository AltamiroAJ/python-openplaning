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

# Experimental, opt-in FSD-consistent interpretation (28C.5).
from .frame_resolved import (
    FrameResolvedComparison, assemble_lewandowski_2004_fsd,
    from_resolved_planing_boat_fsd, compare_legacy_and_fsd,
)
__all__ += [
    'FrameResolvedComparison', 'assemble_lewandowski_2004_fsd',
    'from_resolved_planing_boat_fsd', 'compare_legacy_and_fsd',
]


# Opt-in eigenbranch continuation and Ixz sensitivity (28C.6).
from .modal_tracking import (
    ModalSweep, DominanceCrossing, track_oscillatory_modes,
    sweep_ixz_fsd, refine_dominance_crossings, analyze_ixz_fsd,
)
__all__ += [
    'ModalSweep', 'DominanceCrossing', 'track_oscillatory_modes',
    'sweep_ixz_fsd', 'refine_dominance_crossings', 'analyze_ixz_fsd',
]

# Read-only source-equation audit; does not change either transverse solver (28C.7).
from .source_audit import SourceRollAudit, audit_roll_sources
__all__ += ['SourceRollAudit', 'audit_roll_sources']

# 28C.8: source-definition/equilibrium audit only; does not change solvers.
from .equilibrium_audit import (
    SourceDecomposition, EquilibriumDerivativeTest,
    decompose_lateral_discrepancy, test_equilibrium_derivative,
)
__all__ += [
    'SourceDecomposition', 'EquilibriumDerivativeTest',
    'decompose_lateral_discrepancy', 'test_equilibrium_derivative',
]
