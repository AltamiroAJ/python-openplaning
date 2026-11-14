"""28C.7: non-mutating SOURCE audit of 1996 Eq.(27)-(29) and 2004 Tables 6.2/6.3.

Diagnostics only: no assumptions here authorize substituting coefficients in a
validated stability prediction. All dimensional inputs in SI; angles at API
boundary in degrees, perturbations and derivatives in radians.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, radians, sin, tan

import numpy as np

from .lewandowski import LewandowskiState, RollForceModel


@dataclass(frozen=True)
class SourceRollAudit:
    """All force derivatives N/rad and moment derivatives Nm/rad."""
    side_force_zero_N: float
    side_force_rate_N_per_rad: float
    side_force_rate_five_point_N_per_rad: float
    side_force_rate_2004_table62_N_per_rad: float
    y_phi_from_1996_force_rate_N_per_rad: float
    y_phi_full_projection_N_per_rad: float
    y_phi_2004_table63_N_per_rad: float
    y_phi_1996_projection_minus_1996_rate_N_per_rad: float
    y_phi_projection_residual_N_per_rad: float
    y_phi_expected_geometric_term_N_per_rad: float
    k_phi_dynamic_1996_Nm_per_rad: float
    k_phi_2004_table62_Nm_per_rad: float
    n_phi_full_projection_Nm_per_rad: float
    n_phi_2004_table63_Nm_per_rad: float
    k_phi_static_2004_Nm_per_rad: float
    k_phi_static_legacy_Nm_per_rad: float
    delta_static_lift_2004_N: float
    fsw_2004: float
    side_wetting_condition: str
    lambda_geometry_residual: float
    source_consistency_flags: tuple[str, ...]


def _five_point(f, h: float) -> float:
    return (f(-2*h) - 8*f(-h) + 8*f(h) - f(2*h)) / (12*h)


def _source_fsw(state: LewandowskiState) -> tuple[float, str]:
    """Lewandowski (2004), Table 6.2, with U^2 threshold and L_C2 branch.

    Printed condition: U^2 > g B (lambda - 0.16 tan(beta)/tan(tau)) / (3 sin(tau)).
    Compare dimensionally consistent U^2 in m²/s²; no imperial quantities.
    """
    b = state.beam
    beta, tau = radians(state.beta_deg), radians(state.tau_deg)
    lam = state.lambda_w
    threshold = state.g*b*(lam - .16*tan(beta)/tan(tau))/(3*sin(tau))
    if state.speed**2 > threshold:
        return 1.0, 'high_speed_fsw_equal_one'
    return 1.0 + .603*state.L_C2/(lam*b), 'low_speed_fsw_includes_L_C2'


def _source_static_2004(state: LewandowskiState) -> tuple[float, float, float, str]:
    """Lewandowski (2004) Tables 6.2/6.3, static part of K_phi.

    Source Delta = 0.25*rho*g*lambda**2*sin(2*tau)*B**3.
    Source KB = B*tan(beta)*(0.5 + L_C/L_K)/6.
    """
    lam, b = state.lambda_w, state.beam
    tau, beta = radians(state.tau_deg), radians(state.beta_deg)
    delta = .25*state.rho*state.g*lam**2*sin(2*tau)*b**3
    KB = b*tan(beta)*(.5 + state.L_C/state.L_K)/6
    fsw, regime = _source_fsw(state)
    static = .624*fsw*(-state.rho*state.g*b**3*(state.L_K+3*state.L_C)/48
                       +(state.vcg-KB)*delta)
    return static, delta, fsw, regime


def audit_roll_sources(state: LewandowskiState, *, step_rad: float = 2e-5) -> SourceRollAudit:
    """Compare 1996 side-force calculation with *printed* 2004 Table 6.2.

    2004 F_phi is evaluated directly from Table 6.2 (tau in radians),
    independently of the 1996 Eq.27 finite difference F_phi_1996.
    Compare also the exact derivative of the selected 1996 signed Y(phi):
       dY/dphi = 2 F_phi_1996 sin(beta) + 2 F_1996(0) cos(beta).
    Because 2004 Table 6.3 uses Y_phi=2*F_phi_2004*sin(beta), both the
    normal-force regression and changing projection can contribute differences.
    The meaning of Y(phi) across sources remains physically unresolved.
    """
    if not isinstance(state, LewandowskiState):
        raise TypeError('state must be LewandowskiState')
    if not isfinite(step_rad) or not 0 < step_rad <= .002:
        raise ValueError('step_rad must be in (0,0.002]')
    model = RollForceModel(state, source_variant='lew1996_eq27', beam_mode='constant')
    beta = radians(state.beta_deg)
    f = lambda phi: model.side_normal_force(state.beta_deg + np.rad2deg(phi))
    F0 = f(0.)
    Fphi = _five_point(f, step_rad)
    Fphi_central = (f(step_rad)-f(-step_rad))/(2*step_rad)
    lam = state.lambda_w
    tau = radians(state.tau_deg)
    # Lewandowski (2004), Table 6.2, printed F_phi.  Tau IN RADIANS.
    Fphi2004 = (-(.5*state.rho*state.speed**2*state.beam**2)
                /(4*cos(beta)**2)
                * (((1-sin(beta))*pi*lam/(1+lam)+1.33/pi)*tau
                   +(1-sin(beta))/(cos(beta)*2*(1+lam)**2)))
    Y2004 = 2*Fphi2004*sin(beta)
    Yfrom1996rate = 2*Fphi*sin(beta)
    Yfull = float(_five_point(lambda phi: model.forces_at_heel(phi)[0], step_rad))
    K1996 = float(_five_point(lambda phi: model.forces_at_heel(phi)[1], step_rad))
    K2004 = 2*model.lever_arm()*Fphi2004
    Nfull = Yfull*(state.lcp-state.lcg)
    N2004 = Y2004*(state.lcp-state.lcg)
    Kstatic, delta, fsw, regime = _source_static_2004(state)
    Klegacy = model.static_roll_derivative()
    geom = 2*F0*cos(beta)
    flags = []
    if abs(state.lambda_geometry_residual) > 1e-6:
        flags.append('Savitsky L_C differs from the 1996 Eq.26a L_C assumed for side-force derivative')
    if abs(Yfull-Y2004) > 1e-8*max(1.,abs(Yfull),abs(Y2004)):
        flags.append('1996 full Y(phi) derivative differs from direct 2004 Table 6.2/6.3 Y_phi; regression and projection both contribute')
    if abs(Kstatic-Klegacy) > 1e-8*max(1.,abs(Kstatic),abs(Klegacy)):
        flags.append('legacy static restoring is not identical to 2004 Tables 6.2/6.3')
    return SourceRollAudit(
        float(F0), float(Fphi_central), float(Fphi), float(Fphi2004),
        float(Yfrom1996rate),float(Yfull),float(Y2004),
        float(Yfull-Yfrom1996rate),float(Yfull-Y2004),float(geom),float(K1996),float(K2004),
        float(Nfull),float(N2004),float(Kstatic),float(Klegacy),
        float(delta),float(fsw),regime,float(state.lambda_geometry_residual),tuple(flags))
