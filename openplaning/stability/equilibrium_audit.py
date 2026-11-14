"""28C.8: distinguish equilibrium residuals, heel derivatives and wet-length choices.

Diagnostic only. The published scalar 'static lift' does NOT specify a lateral
static-force derivative. Equilibrium Y(0)=0 does not imply Y_phi(0)=0.
No stability solver is changed; SI except source empirical angular conventions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, radians, sin, tan

import numpy as np

from .lewandowski import LewandowskiState, RollForceModel
from .source_audit import audit_roll_sources


def _table_2004_force_rate(state: LewandowskiState, lam: float) -> float:
    """Printed Lewandowski 2004 Table 6.2 F_phi, varying ONLY lambda.

    This is a *counterfactual* evaluation, not a new hydrodynamic closure.
    """
    if not isfinite(lam) or lam <= 0:
        raise ValueError('mean wetted length ratio must be positive and finite')
    beta, tau = radians(state.beta_deg), radians(state.tau_deg)
    return (-(.5*state.rho*state.speed**2*state.beam**2)
            /(4*cos(beta)**2)
            * (((1-sin(beta))*pi*lam/(1+lam)+1.33/pi)*tau
               +(1-sin(beta))/(cos(beta)*2*(1+lam)**2)))


@dataclass(frozen=True)
class SourceDecomposition:
    """Additive identity in N/rad, using stated sequence of counterfactuals.

    R = Y96_full - Y04(lambda_Sav) = projection + source + lambda.
    The splitting is path-dependent; not causal identification without a common
    physical closure for both sources.
    """
    lambda_sav: float
    lambda_eq26: float
    lc_sav_m: float
    lc_eq26_m: float
    y1996_full: float
    y1996_normal_rate: float
    y2004_at_eq26: float
    y2004_at_sav: float
    projection_term: float
    source_difference_at_matched_lambda: float
    wet_length_difference: float
    total_difference: float
    recomposed_difference: float
    n_difference: float
    flags: tuple[str, ...]


def decompose_lateral_discrepancy(state: LewandowskiState, *, step_rad=2e-5) -> SourceDecomposition:
    """Separate 1996 projection, source expression and L_C conventions.

    1996 Eq.26a recomputes L_C(phi) with L_K fixed regardless of the
    `state.L_C` supplied by OpenPlaning; hence changing state.L_C in isolation
    does not change the 1996 per-side normal force at fixed L_K. The 2004
    coefficient instead uses lambda=(L_K+state.L_C)/(2b). The decomposition
    uses a single *declared path*, not an assumed unique causal partition.
    """
    if not isinstance(state, LewandowskiState):
        raise TypeError('expected LewandowskiState')
    src = audit_roll_sources(state, step_rad=step_rad)
    lam_sav = state.lambda_w
    lam_eq = state.lambda_eq26
    if lam_eq <= 0 or not isfinite(lam_eq):
        raise ValueError('Eq.26a implies nonpositive lambda: decomposition invalid')
    y04_sav = 2*_table_2004_force_rate(state, lam_sav)*sin(radians(state.beta_deg))
    y04_eq = 2*_table_2004_force_rate(state, lam_eq)*sin(radians(state.beta_deg))
    projection = src.y_phi_full_projection_N_per_rad - src.y_phi_from_1996_force_rate_N_per_rad
    source = src.y_phi_from_1996_force_rate_N_per_rad - y04_eq
    wet = y04_eq - y04_sav
    total = src.y_phi_full_projection_N_per_rad - y04_sav
    parts = projection + source + wet
    if not np.isclose(total, parts, rtol=1e-11, atol=1e-7):
        raise RuntimeError('decomposition identity failed')
    flags = (
        '1996 per-side L_C is recomputed from Eq.26a, not state.L_C',
        '2004 Table 6.2 uses lambda from state.L_C',
        'Additive allocation depends on counterfactual order',
        'No physical lateral static-force derivative is implied by scalar static lift',
    )
    return SourceDecomposition(
        lam_sav, lam_eq, state.L_C,
        state.L_K-state.beam*tan(radians(state.beta_deg))/(pi*tan(radians(state.tau_deg))),
        src.y_phi_full_projection_N_per_rad,
        src.y_phi_from_1996_force_rate_N_per_rad,
        y04_eq, y04_sav, projection, source, wet, total, parts,
        total*(state.lcp-state.lcg), flags,
    )


@dataclass(frozen=True)
class EquilibriumDerivativeTest:
    """Force and Jacobian at phi=0 for user-defined opposing-force slope.

    At equilibrium, an opposing force can have the same value -Y(0) but an
    arbitrary heel derivative. Since a symmetric 1996 two-side model has
    Y(0)=0, constant balancing forces do not remove its slope.
    """
    y_dynamic_zero_N: float
    y_dynamic_rate_N_per_rad: float
    static_force_zero_for_balance_N: float
    prescribed_static_rate_N_per_rad: float
    total_force_zero_N: float
    total_rate_N_per_rad: float
    geometric_rate_N_per_rad: float
    cancellation_requires_rate_N_per_rad: float


def test_equilibrium_derivative(state: LewandowskiState, *,
                                static_heel_rate_N_per_rad=0., step_rad=2e-5) -> EquilibriumDerivativeTest:
    """Evaluate equilibrium and linearization separately.

    The prescribed static rate is an EXTERNAL hypothesis (default zero), not
    a derived Lewandowski static lateral force. Setting it equal to
    -geometric_rate cancels only that term and does NOT remove the normal
    force-rate contribution. Equilibrium alone cannot determine this rate.
    """
    rate = float(static_heel_rate_N_per_rad)
    if not isfinite(rate):
        raise ValueError('static_heel_rate_N_per_rad must be finite')
    audit = audit_roll_sources(state, step_rad=step_rad)
    model = RollForceModel(state, source_variant='lew1996_eq27', beam_mode='constant')
    y0 = float(model.forces_at_heel(0.)[0])
    counter0 = -y0
    return EquilibriumDerivativeTest(
        y0, audit.y_phi_full_projection_N_per_rad, counter0,
        rate, y0+counter0, audit.y_phi_full_projection_N_per_rad+rate,
        audit.y_phi_expected_geometric_term_N_per_rad,
        -audit.y_phi_expected_geometric_term_N_per_rad,
    )
