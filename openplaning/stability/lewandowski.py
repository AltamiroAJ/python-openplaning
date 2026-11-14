"""Traceable Lewandowski (1996) port/starboard force approximation (SI).

Scope: static and dynamic *roll-angle derivatives* at phi=0, without
computing a transverse stability eigenvalue. This is an intentionally small,
non-mutating replacement for the roll-force portion of dynroll.

Primary source: E.M. Lewandowski, "The Transverse Dynamic Stability of
Hard-Chine Planing Craft" (1996), eqs (25)-(29), Fig. 3; Appendix for
static restoring; 2004 monograph Table 6.2. The 1996 Eq. (27) coefficient
1.33/4 is distinct from 1/3 coded in the unfinished dynroll branch.

Frame: x forward from transom; y port; z up; phi, derivatives in radians.
Signed Y/K/N are defined here exactly as in the dynroll port-minus-starboard
implementation, not asserted to equal a published global coordinate frame.

Inputs: SI rho [kg/m3], g [m/s2], U [m/s], b,Lk,Lc,LC2,KG,LCG,LCP [m].
Angles beta_deg,tau_deg are OpenPlaning degrees, converted internally.
This module deliberately does not call get_steady_trim or change PlaningBoat.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import cos, isfinite, pi, sin, tan
from typing import Literal
import numpy as np

_COEFF = {"lew1996_eq27": 1.33 / 4.0, "dynroll_1997_prototype": 1.0 / 3.0}


@dataclass(frozen=True)
class LewandowskiState:
    speed: float
    rho: float
    g: float
    beam: float
    beta_deg: float
    tau_deg: float
    L_K: float
    L_C: float
    L_C2: float
    vcg: float
    lcg: float
    lcp: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if not isfinite(float(val)):
                raise ValueError(f"{name} must be finite")
        for name in ("speed", "rho", "g", "beam", "L_K"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 < self.beta_deg < 89:
            raise ValueError("deadrise must be in (0,89) degrees")
        if not 0 < self.tau_deg < 45:
            raise ValueError("trim must be in (0,45) degrees")
        if self.L_C < 0 or self.L_C2 < 0 or self.L_C > self.L_K:
            raise ValueError("require 0 <= L_C <= L_K and L_C2 >= 0")
        if self.vcg < 0:
            raise ValueError("vcg must be nonnegative")

    @property
    def lambda_w(self) -> float:
        return (self.L_K + self.L_C) / (2.0 * self.beam)

    @property
    def lambda_eq26(self) -> float:
        """Mean wet-length ratio implied by Lewandowski (1996), Eq. (26a)."""
        beta = np.deg2rad(self.beta_deg)
        tau = np.deg2rad(self.tau_deg)
        return self.L_K/self.beam - tan(beta)/(2*pi*tan(tau))

    @property
    def lambda_geometry_residual(self) -> float:
        """(OpenPlaning lambda) - (1996 Eq 26a lambda); not auto-corrected."""
        return self.lambda_w - self.lambda_eq26


@dataclass(frozen=True)
class RollForceDerivatives:
    """Derivatives per *radian*: Y_phi[N/rad], K_phi[Nm/rad], N_phi[Nm/rad]."""
    Y_phi_dynamic: float
    K_phi_dynamic: float
    N_phi_dynamic: float
    K_phi_static: float
    step_rad: float
    scheme: str
    source_variant: str

    @property
    def K_phi_total(self):
        return self.K_phi_dynamic + self.K_phi_static


class RollForceModel:
    """1996 bottom-normal force split with two explicitly documented variants.

    `source_variant="lew1996_eq27"` follows coefficient 1.33/4 in Eq. (27).
    `"dynroll_1997_prototype"` follows the 1/3 in the unfinished code;
    it is NOT evidence that the 1997 paper adopted that exact coefficient.

    `beam_mode="constant"` uses Eq. (26a) for lambda(beta_eff).
    `"cosine_projected"` imitates dynroll's b_eff expression while keeping
    L_K fixed; it does not call OpenPlaning's geometry solver and hence is
    NOT the exact dynroll geometry algorithm.
    """
    def __init__(
        self,
        state: LewandowskiState,
        *,
        source_variant: Literal["lew1996_eq27", "dynroll_1997_prototype"] = "lew1996_eq27",
        beam_mode: Literal["constant", "cosine_projected"] = "constant",
    ):
        if not isinstance(state, LewandowskiState):
            raise TypeError("state must be a LewandowskiState")
        if source_variant not in _COEFF:
            raise ValueError("unsupported source_variant")
        if beam_mode not in ("constant", "cosine_projected"):
            raise ValueError("unsupported beam_mode")
        self.state = state
        self.source_variant = source_variant
        self.beam_mode = beam_mode

    def _geometry(self, beta_eff_rad: float):
        s = self.state
        beta = np.deg2rad(s.beta_deg)
        beam_eff = s.beam if self.beam_mode == "constant" else s.beam * cos(beta_eff_rad) / cos(beta)
        if beam_eff <= 0 or not 0 < beta_eff_rad < pi/2:
            raise ValueError("effective deadrise out of range")
        # Eq. (26) assumes L_K is invariant under infinitesimal heel.
        L_C_eff = s.L_K - beam_eff * tan(beta_eff_rad) / (pi * tan(np.deg2rad(s.tau_deg)))
        lam = (s.L_K + L_C_eff) / (2.0 * beam_eff)
        if lam <= 0 or L_C_eff < 0:
            raise ValueError("invalid wetted geometry: lambda or L_C is negative")
        return beam_eff, lam

    def side_normal_force(self, beta_effective_deg: float) -> float:
        """F_n (N), on one bottom side; 1996 Eq. (27), not total lift."""
        s = self.state
        beta = np.deg2rad(float(beta_effective_deg))
        b_eff, lam = self._geometry(beta)
        tau = np.deg2rad(s.tau_deg)
        correction = _COEFF[self.source_variant]
        bracket = (
            pi / 4 * (1.0 - sin(beta)) * cos(tau) * lam / (1.0 + lam)
            + correction * lam * cos(tau) * sin(2.0 * tau) * cos(beta)
        )
        # Outer 1/2: dynamic pressure. Inner 1/2: split to the two sides.
        return 0.5 * s.rho * s.speed**2 * b_eff**2 * (sin(2.0*tau) / (2.0*cos(beta))) * bracket

    def lever_arm(self) -> float:
        """Constant arm h2 [m] from Lewandowski (1996), Eq. (28)."""
        s = self.state
        beta = np.deg2rad(s.beta_deg)
        return 0.8*pi*s.beam/(8*cos(beta)) - s.vcg*sin(beta)

    def forces_at_heel(self, phi_rad: float) -> np.ndarray:
        """Return [Y_d(N), K_d(Nm), N_d(Nm)] at heel in radians.

        Signed projection follows the unfinished dynroll code, for traceability.
        """
        s = self.state
        phi = float(phi_rad)
        if not isfinite(phi):
            raise ValueError("heel must be finite")
        beta = np.deg2rad(s.beta_deg)
        F_port = self.side_normal_force(np.rad2deg(beta + phi))
        F_stbd = self.side_normal_force(np.rad2deg(beta - phi))
        Y = F_port * sin(beta+phi) - F_stbd * sin(beta-phi)
        K = (F_port-F_stbd) * self.lever_arm()
        N = Y*(s.lcp-s.lcg)
        return np.array([Y, K, N], dtype=float)

    def static_roll_derivative(self) -> float:
        """Dynroll Appendix approximation for static restoring K_phi, Nm/rad.

        This is model-specific, not a hydrostatic GZ calculation. It assumes
        lateral symmetry and same simplified side-wetting logic as dynroll.
        """
        s = self.state
        b, Lk, Lc, lam = s.beam, s.L_K, s.L_C, s.lambda_w
        beta, tau = np.deg2rad(s.beta_deg), np.deg2rad(s.tau_deg)
        I_T = b**3*(Lk+3*Lc)/48
        KB = b*tan(beta)*(0.5+Lc/Lk)/6
        Delta_s = (0.25*s.rho*s.g*b**3*lam**2*sin(2*tau)
                   * (1+(Lk-Lc)**2/(3*(Lk+Lc)**2)))
        fsw = 1 + 0.603*s.L_C2/(lam*b) if s.L_C2 > 0 else 1.0
        return 0.624*fsw*(-s.rho*s.g*I_T+(s.vcg-KB)*Delta_s)

    def derivatives(self, *, step_rad: float = 1e-5) -> RollForceDerivatives:
        """Central finite difference with a radian perturbation.

        Explicitly avoids differentiating a degree-valued API as radians.
        The step is configurable for convergence studies.
        """
        if not isfinite(step_rad) or step_rad <= 0 or step_rad > 0.02:
            raise ValueError("step_rad must be finite and in (0,0.02]")
        yp, kp, np_ = ((self.forces_at_heel(step_rad) - self.forces_at_heel(-step_rad))
                         / (2*step_rad))
        return RollForceDerivatives(
            Y_phi_dynamic=float(yp), K_phi_dynamic=float(kp),
            N_phi_dynamic=float(np_), K_phi_static=self.static_roll_derivative(),
            step_rad=step_rad, scheme="central_difference_radians",
            source_variant=self.source_variant,
        )


def from_planing_boat(boat) -> LewandowskiState:
    """Read an already-resolved PlaningBoat state; never mutate or solve it.

    Requires get_steady_trim() and get_forces() to have populated geometric
    lengths and LCP. Does NOT infer LCP or L_C2 from arbitrary assumptions.
    """
    required = ("speed", "rho", "g", "beam", "beta", "tau", "L_K", "L_C",
                "L_C2", "vcg", "lcg", "lcp")
    missing = [key for key in required if not hasattr(boat, key)]
    if missing:
        raise ValueError("PlaningBoat state incomplete; run equilibrium and forces first: " + ", ".join(missing))
    return LewandowskiState(
        speed=float(boat.speed), rho=float(boat.rho), g=float(boat.g),
        beam=float(boat.beam), beta_deg=float(boat.beta), tau_deg=float(boat.tau),
        L_K=float(boat.L_K), L_C=float(boat.L_C), L_C2=float(boat.L_C2),
        vcg=float(boat.vcg), lcg=float(boat.lcg), lcp=float(boat.lcp),
    )
