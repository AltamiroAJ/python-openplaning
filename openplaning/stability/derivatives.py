"""Lewandowski semi-empirical hull coefficients, SI inputs and outputs.

Sources: Lewandowski (1997), Marine Technology 34(2), Appendix p.117;
Lewandowski (2004), The Dynamics of Marine Craft, Tables 6.2 and 6.3,
pp.369-370. The 1997 published K_phi_dot h1*Yv/U term is dimensionally
inconsistent and MUST NOT enter a stability matrix without resolution.

This module provides a stand-alone, non-mutating evaluation of the *hull*
coefficients only. It excludes appendages, inertial terms and restoring
coefficients. All physical input/output quantities use SI. Beta and tau
are in degrees only at the OpenPlaning API boundary and where explicitly
required by empirical regression definitions.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, radians, sin, tan, sqrt
from typing import Literal


@dataclass(frozen=True)
class TransverseCoefficientState:
    speed: float       # m/s
    rho: float         # kg/m^3
    g: float           # m/s^2
    beam: float        # m
    L_K: float         # m
    L_C: float         # m
    draft_transom: float  # m (T, source: PlaningBoat.T)
    beta_deg: float    # degrees in fitted regressions
    tau_deg: float     # degrees in fitted regressions
    vcg: float         # m
    lcg: float         # m

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in ('speed', 'rho', 'g', 'beam', 'L_K', 'draft_transom'):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0 <= self.L_C <= self.L_K):
            raise ValueError('require 0 <= L_C <= L_K')
        if not (0 < self.beta_deg <= 45):
            raise ValueError('deadrise outside 0 < beta <= 45 degrees')
        if not (0 < self.tau_deg < 45):
            raise ValueError('require 0 < tau < 45 degrees')
        if self.vcg < 0 or self.lcg < 0:
            raise ValueError('CG positions must be nonnegative')

    @property
    def lambda_w(self):
        return (self.L_K + self.L_C) / (2*self.beam)

    @property
    def cv(self):
        return self.speed / sqrt(self.g*self.beam)

    @property
    def t_ratio(self):
        return self.draft_transom / self.beam


COEFFICIENT_UNITS = {
    'Y_v_dot': 'kg', 'K_v_dot': 'kg*m', 'N_v_dot': 'kg*m',
    'Y_r_dot': 'kg*m', 'K_r_dot': 'kg*m^2', 'N_r_dot': 'kg*m^2',
    'Y_phi_ddot': 'kg*m', 'K_phi_ddot': 'kg*m^2', 'N_phi_ddot': 'kg*m^2',
    'Y_v': 'kg/s', 'K_v': 'kg*m/s', 'N_v': 'kg*m/s',
    'Y_r': 'kg*m/s', 'K_r': 'kg*m^2/s', 'N_r': 'kg*m^2/s',
    'Y_phi_dot': 'kg*m/s', 'K_phi_dot': 'kg*m^2/s',
    'N_phi_dot': 'kg*m^2/s',
}


@dataclass(frozen=True)
class HullDerivatives:
    source: str
    state: TransverseCoefficientState
    values: dict[str, float]
    units: dict[str, str]
    dimensional_warnings: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def dimensionally_consistent(self) -> bool:
        return not self.dimensional_warnings

    def require_dimensional_consistency(self) -> 'HullDerivatives':
        if self.dimensional_warnings:
            raise ValueError('dimensionally unresolved published variant: ' +
                             '; '.join(self.dimensional_warnings))
        return self

    def __getitem__(self, key):
        return self.values[key]


def evaluate_hull_derivatives(
    state: TransverseCoefficientState,
    *,
    source: Literal['lew2004_table63', 'lew1997_appendix', 'dynroll_prototype'] = 'lew2004_table63',
) -> HullDerivatives:
    """Return hull-only derivatives from cited tables, without inertial mass.

    The 1997 and dynroll variants are diagnostics only (see dimensional_warnings).
    The source prints beta**0.6 and tau+5.28 with ANGLES IN DEGREES.
    Ordinary trigonometric functions instead use radians.

    Warning: the 1997 printed K_phi_dot = base + h1*Yv/U is not
    dimensionally additive: [h1*Yv/U] = kg*m, but base = kg*m^2/s.
    The 2004 version uses h1*Yv and is dimensionally coherent.
    """
    if not isinstance(state, TransverseCoefficientState):
        raise TypeError('state must be TransverseCoefficientState')
    if source not in ('lew2004_table63', 'lew1997_appendix', 'dynroll_prototype'):
        raise ValueError('unsupported source variant')
    s = state
    b, lk, lc, rho, U = s.beam, s.L_K, s.L_C, s.rho, s.speed
    beta, beta_rad, tau = s.beta_deg, radians(s.beta_deg), s.tau_deg
    t = s.t_ratio
    lam, cv = s.lambda_w, s.cv
    L = (lk + lc)/2  # Source symbol L; explicitly chosen as mean wetted length.
    kbeta = .06641 + .00716*beta + .0003861*beta**2
    h1 = (s.vcg - .3927*b*tan(beta_rad))*(s.vcg - .306*b)

    Y_v_dot = -b*b*rho*tan(beta_rad)*kbeta*(lk+2*lc)/12
    N_v_dot = -b*b*rho*tan(beta_rad)*kbeta*(lk**2+2*lk*lc+3*lc**2)/48
    N_r_dot = -b*b*rho*tan(beta_rad)*kbeta*(lk**3+2*lk**2*lc+3*lc**2*lk+4*lc**3)/120
    K_phi_ddot = -.010237*rho*b**5*lam*(1-sin(beta_rad))+h1*Y_v_dot

    Y_v = -.5*rho*U*b**2*(.6494*beta**.6*t**2*cv**2)
    K_v = Y_v*(-s.vcg+1.5145*b/beta**.342)
    N_v = Y_v*(-s.lcg+12.384*b*t**.45/(tau+5.28))
    Y_r = .5*rho*U*b**2*L*(55.439*beta**.6*t**3)*(.02754-.5949*t/(tau+5.28))

    if source == 'lew2004_table63':
        N_r = .5*rho*U*b**3*L*(73.918*beta**.6*t**3)*(.00638+6.714*t**2/(tau+5.28)**2)
    else:
        N_r = -.5*rho*U*b**3*L*(73.918*beta**.6*t**3)*(.00638-.08289*t**2/(tau+5.28)**2)

    K_phi_dot_base = -(1-sin(beta_rad))*(.029*cv+.02*lam)*rho*s.g*b**4*sqrt(b/s.g)
    if source == 'lew1997_appendix':
        K_phi_dot = K_phi_dot_base + h1*Y_v/U
    else:
        K_phi_dot = K_phi_dot_base + h1*Y_v

    # The unfinished dynroll prototype differs from both publications: N_r_dot
    # uses coefficient 2 for (L_C^2 L_K), while both printed tables use 3.
    # It also omits C_V^2 in Y_v and uses a different expression for N_r.
    if source == 'dynroll_prototype':
        N_r_dot = -b*b*rho*tan(beta_rad)*kbeta*(lk**3+2*lk**2*lc+2*lc**2*lk+4*lc**3)/120
        Y_v = -.5*rho*U*b**2*(.6494*beta**.6*t**2)
        K_v = Y_v*(-s.vcg+1.5145*b/beta**.342)
        N_v = Y_v*(-s.lcg+12.384*b*t**.45/(tau+5.28))
        K_phi_dot = K_phi_dot_base + h1*Y_v

    result = dict(
        Y_v_dot=Y_v_dot, K_v_dot=0.0, N_v_dot=N_v_dot,
        Y_r_dot=N_v_dot, K_r_dot=0.0, N_r_dot=N_r_dot,
        Y_phi_ddot=0.0, K_phi_ddot=K_phi_ddot, N_phi_ddot=0.0,
        Y_v=Y_v, K_v=K_v, N_v=N_v, Y_r=Y_r, K_r=0.0, N_r=N_r,
        Y_phi_dot=0.0, K_phi_dot=K_phi_dot, N_phi_dot=0.0,
    )
    if not all(isfinite(v) for v in result.values()):
        raise ValueError('computed non-finite derivative')
    warnings = ()
    if source == 'lew1997_appendix':
        warnings = ('1997 K_phi_dot: h1*Y_v/U has kg*m versus kg*m^2/s for the base term',)
    if source == 'dynroll_prototype':
        warnings = ('dynroll is an incomplete unpublished prototype; not a published coefficient set',)
    return HullDerivatives(
        source=source, state=s, values=result,
        units=dict(COEFFICIENT_UNITS), dimensional_warnings=warnings,
        notes=(
            'HULL ONLY; appendage terms excluded',
            'Beta and tau regression inputs are in degrees; trigonometric operations use radians',
            'L is taken as (L_K+L_C)/2; this is an explicit model choice',
            'T is transom draft supplied by PlaningBoat, not inferred from L_K*sin(tau)',
            'The sign convention must be reconciled with the assembled EOM before stability analysis',
        ),
    )


def coefficient_state_from_planing_boat(boat) -> TransverseCoefficientState:
    """Read required *resolved* state without modifying or re-solving boat."""
    required = ('speed', 'rho', 'g', 'beam', 'L_K', 'L_C', 'T',
                'beta', 'tau', 'vcg', 'lcg')
    missing = [k for k in required if not hasattr(boat, k)]
    if missing:
        raise ValueError('PlaningBoat geometry/equilibrium missing: '+', '.join(missing))
    return TransverseCoefficientState(
        speed=float(boat.speed), rho=float(boat.rho), g=float(boat.g),
        beam=float(boat.beam), L_K=float(boat.L_K), L_C=float(boat.L_C),
        draft_transom=float(boat.T), beta_deg=float(boat.beta),
        tau_deg=float(boat.tau), vcg=float(boat.vcg), lcg=float(boat.lcg),
    )
