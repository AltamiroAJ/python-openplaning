"""Traceable *linear* sway--roll--yaw dynamics: Lewandowski (2004), 6.8--6.12.

DO NOT interpret the numerical eigenvalues as validated sea-trial predictions.
The 1996 port/starboard force implementation retains experimental sign and
wetted-geometry conventions; appendages are not implemented. Internal SI.

State x = [v (m/s), phi (rad), p=phi_dot (rad/s), r=yaw rate (rad/s)].
F = [Y (N), K (Nm), N (Nm)]; x forward, y port, z up for the mass model.

The source 'Ixz' in 2004 Eq. 6.8 is the *raw* integral(x*z dm), i.e. negative
of the tensor entry I[0,2] in the adopted right-handed body frame. This
mapping is an explicit convention to be reconciled against the book's axes
before physical interpretation. Do not replace it with tensor[0,2].
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import isfinite, pi

import numpy as np
from numpy.polynomial import Polynomial

from .derivatives import HullDerivatives, evaluate_hull_derivatives, coefficient_state_from_planing_boat
from .lewandowski import RollForceDerivatives, RollForceModel, from_planing_boat


@dataclass(frozen=True)
class RigidInertia:
    """Rigid *physical* mass inertia about CG, kg, kg*m², in body axes."""
    mass: float
    Ixx: float
    Izz: float
    Ixz_tensor: float
    Ixy_tensor: float = 0.0
    Iyz_tensor: float = 0.0

    def __post_init__(self):
        values = (self.mass, self.Ixx, self.Izz, self.Ixz_tensor,
                  self.Ixy_tensor, self.Iyz_tensor)
        if not all(isfinite(float(x)) for x in values):
            raise ValueError('rigid inertia values must be finite')
        if min(self.mass, self.Ixx, self.Izz) <= 0:
            raise ValueError('mass, Ixx and Izz must be positive')
        if abs(self.Ixz_tensor) > np.sqrt(self.Ixx * self.Izz) * (1 + 1e-12):
            raise ValueError('Ixz exceeds the Cauchy bound for a physical inertia tensor')

    @classmethod
    def from_snapshot(cls, snapshot, *, symmetry_tolerance: float = 1e-8) -> 'RigidInertia':
        """Read a MassResult or the snapshot from boat.get_mass_properties_snapshot().

        No alteration to snapshot or boat. Reject transverse asymmetry in
        the current *symmetric* hydrodynamic/equilibrium formulation.
        """
        if not isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
            raise ValueError('symmetry_tolerance must be nonnegative and finite')
        M = float(snapshot.mass)
        I = np.asarray(snapshot.inertia_tensor, dtype=float)
        cg = np.asarray(snapshot.center_of_gravity, dtype=float)
        if not isfinite(M) or M <= 0 or I.shape != (3, 3) or cg.shape != (3,):
            raise ValueError('invalid mass snapshot')
        if not np.all(np.isfinite(I)) or not np.all(np.isfinite(cg)):
            raise ValueError('mass properties must be finite')
        if not np.allclose(I, I.T, rtol=1e-10, atol=1e-9):
            raise ValueError('inertia tensor must be symmetric')
        # Relative to CG, symmetry imposes Ixy=Iyz=0 and transverse CG=0.
        # The tolerance uses reference gyration length sqrt(tr(I)/M).
        scale = max(np.sqrt(max(np.trace(I), 0.0) / M), 1e-9)
        if abs(cg[1]) > symmetry_tolerance * scale:
            raise ValueError('lateral CG violates the symmetric transverse model')
        inertia_scale = max(float(np.trace(I)), M * scale**2, 1e-12)
        if max(abs(I[0, 1]), abs(I[1, 2])) > symmetry_tolerance * inertia_scale:
            raise ValueError('Ixy or Iyz indicates an asymmetric inertia tensor')
        if min(I[0, 0], I[2, 2]) <= 0:
            raise ValueError('roll and yaw inertia must be strictly positive')
        if np.linalg.eigvalsh(I)[0] < -1e-9 * inertia_scale:
            raise ValueError('inertia tensor must be positive semidefinite')
        return cls(M, float(I[0, 0]), float(I[2, 2]), float(I[0, 2]),
                   float(I[0, 1]), float(I[1, 2]))

    @property
    def Ixz_book_raw(self) -> float:
        """Integral(x*z dm) = -I_xz_tensor, assuming aligned body axes."""
        return -self.Ixz_tensor


@dataclass(frozen=True, eq=False)
class LinearTransverseSystem:
    """The descriptor equation M*qdot + D*q + h*phi = 0; q=[v,p,r]."""
    acceleration_matrix: np.ndarray
    rate_matrix: np.ndarray
    heel_column: np.ndarray
    state_matrix: np.ndarray
    characteristic_polynomial: np.ndarray  # descending powers s^4...s^0
    eigenvalues: np.ndarray
    spectral_abscissa: float  # 1/s
    classification: str
    tolerance_per_second: float
    assumptions: tuple[str, ...]

    @property
    def oscillation_frequencies_hz(self) -> np.ndarray:
        return abs(self.eigenvalues.imag) / (2 * pi)

    @property
    def growth_rates_per_second(self) -> np.ndarray:
        return self.eigenvalues.real.copy()


def _matrix(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite and of shape {shape}')
    return a.copy()


def _determinant_polynomial(M: np.ndarray, D: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Independent determinant of 2004 Eq 6.12 in s, without Eq 6.14.

    Columns are v, phi, yaw_rate; the second column is quadratic in s.
    """
    a = [[Polynomial([D[i, 0], M[i, 0]]),
          Polynomial([h[i], D[i, 1], M[i, 1]]),
          Polynomial([D[i, 2], M[i, 2]])] for i in range(3)]
    det = (a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
           - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
           + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]))
    coef = np.zeros(5)
    raw = det.coef
    if len(raw) > 5:
        if np.max(np.abs(raw[5:])) > 1e-10 * max(1.0, np.max(abs(raw))):
            raise RuntimeError('unexpected polynomial degree')
    coef[-min(5, len(raw)):] = raw[:5][::-1]
    return coef


def _match_roots(a, b, *, tolerance=1e-5) -> bool:
    """Small independent root correspondence (four modes; no SciPy needed)."""
    scale = max(1., max(np.abs(a)), max(np.abs(b)))
    return any(max(abs(a[i]-b[j]) for i, j in enumerate(order))
               < tolerance * scale for order in permutations(range(4)))


def linearize_transverse_system(
    acceleration_matrix, rate_matrix, heel_column, *,
    mass: float, beam: float, tolerance_per_second: float = 1e-8,
    assumptions: tuple[str, ...] = (),
) -> LinearTransverseSystem:
    """Numerical linearization of the 2004 descriptor form, no model assumptions.

    M*qdot + D*[v,p,r] + h*phi = 0. Designed for independent matrix tests.
    Row units: [N, Nm, Nm]; columns qdot units [m/s², rad/s², rad/s²].
    For singularity checking, the mixed units are nondimensionalized by
    (mass, mass*beam, mass*beam) rows and (1, 1/beam, 1/beam) columns.
    """
    M = _matrix(acceleration_matrix, (3, 3), 'acceleration_matrix')
    D = _matrix(rate_matrix, (3, 3), 'rate_matrix')
    h = _matrix(heel_column, (3,), 'heel_column')
    if not isfinite(mass) or mass <= 0 or not isfinite(beam) or beam <= 0:
        raise ValueError('mass and beam must be positive and finite')
    if not isfinite(tolerance_per_second) or tolerance_per_second < 0:
        raise ValueError('tolerance_per_second must be finite and nonnegative')
    # Fix scale explicitly: do not assess conditioning on the raw mixed-units M.
    scaled = (np.diag([1/mass, 1/(mass*beam), 1/(mass*beam)])
              @ M @ np.diag([1, 1/beam, 1/beam]))
    condition = float(np.linalg.cond(scaled))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f'ill-conditioned or singular scaled acceleration matrix: {condition:.3g}')
    Q = np.column_stack((D[:, 0], h, D[:, 1], D[:, 2]))
    A = np.zeros((4, 4), dtype=float)
    A[1, 2] = 1.
    A[[0, 2, 3], :] = -np.linalg.solve(M, Q)
    eigenvalues = np.linalg.eigvals(A)
    polynomial = _determinant_polynomial(M, D, h)
    # Cross-check via *independently constructed* characteristic polynomial.
    poly_state = np.poly(A)
    if abs(polynomial[0]) <= 1e-14 * max(abs(polynomial)):
        raise ValueError('characteristic polynomial has deficient degree')
    p = polynomial / polynomial[0]
    if not np.allclose(p, poly_state, rtol=5e-7, atol=1e-6):
        raise RuntimeError('determinant polynomial disagrees with state-space polynomial')
    if not _match_roots(np.roots(polynomial), eigenvalues):
        raise RuntimeError('determinant and state-space eigenvalues disagree')
    max_growth = float(np.max(eigenvalues.real))
    if np.any(eigenvalues.real > tolerance_per_second):
        classification = 'unstable_linear_model'
    elif np.all(eigenvalues.real < -tolerance_per_second):
        classification = 'stable_linear_model'
    else:
        classification = 'marginal_or_unresolved'
    for x in (M, D, h, A, polynomial, eigenvalues):
        x.setflags(write=False)
    return LinearTransverseSystem(M, D, h, A, polynomial, eigenvalues,
                                 max_growth, classification, tolerance_per_second,
                                 tuple(assumptions))


def _compare_states(hull: HullDerivatives, roll_state, *, tolerance=1e-9):
    s, t = hull.state, roll_state
    pairs = [('speed', 'speed'), ('rho', 'rho'), ('g', 'g'),
             ('beam', 'beam'), ('L_K', 'L_K'), ('L_C', 'L_C'),
             ('vcg', 'vcg'), ('lcg', 'lcg'),
             ('beta_deg', 'beta_deg'), ('tau_deg', 'tau_deg')]
    for a, b in pairs:
        x, y = float(getattr(s, a)), float(getattr(t, b))
        if not np.isclose(x, y, rtol=tolerance, atol=tolerance):
            raise ValueError(f'inconsistent source states for {a}: {x} vs {y}')


def assemble_lewandowski_2004(
    hull: HullDerivatives,
    roll: RollForceDerivatives,
    rigid: RigidInertia,
    *,
    roll_state,
    tolerance_per_second: float = 1e-8,
) -> LinearTransverseSystem:
    """2004 Eqs 6.8--6.12 for *hull only*, aligned axis/sign hypothesis.

    Enforce 2004 dimensional variant and published 1996 Eq 27 roll force.
    The roll Y_phi, N_phi sign remains unverified across source axes. The
    output classification is mathematical, NOT a physical safety verdict.
    """
    if not isinstance(hull, HullDerivatives) or not isinstance(roll, RollForceDerivatives):
        raise TypeError('hull and roll must be source-labeled derivative results')
    if not isinstance(rigid, RigidInertia):
        raise TypeError('rigid must be RigidInertia')
    if hull.source != 'lew2004_table63':
        raise ValueError('only 2004 source is allowed in the assembly')
    hull.require_dimensional_consistency()
    if roll.source_variant != 'lew1996_eq27':
        raise ValueError('roll source must be the published 1996 Eq 27 variant')
    _compare_states(hull, roll_state)
    if rigid.mass <= 0 or min(rigid.Ixx, rigid.Izz) <= 0:
        raise ValueError('invalid rigid mass and inertia')
    v = hull.values
    yphi = roll.Y_phi_dynamic
    kphi = roll.K_phi_total
    nphi = roll.N_phi_dynamic
    U = hull.state.speed
    Ixz = rigid.Ixz_book_raw
    M = np.array([
        [v['Y_v_dot'] - rigid.mass, v['Y_phi_ddot'], v['Y_r_dot']],
        [v['K_v_dot'], v['K_phi_ddot'] - rigid.Ixx, v['K_r_dot'] + Ixz],
        [v['N_v_dot'], v['N_phi_ddot'] + Ixz, v['N_r_dot'] - rigid.Izz],
    ])
    D = np.array([
        [v['Y_v'], v['Y_phi_dot'], v['Y_r'] - rigid.mass*U],
        [v['K_v'], v['K_phi_dot'], v['K_r']],
        [v['N_v'], v['N_phi_dot'], v['N_r']],
    ])
    h = np.array([yphi, kphi, nphi])
    assumptions = (
        'Lewandowski 2004 Eqs 6.8-6.12, hull-only; no appendages',
        'Roll forces from 1996 Eq 27, with dynamical sign convention still under review',
        'Book Ixz treated as raw integral(x*z dm)=-physical inertia tensor[0,2]',
        'Body axes, longitudinal speed U and input equilibrium treated as fixed',
        'No assertion that eigenvalue signs predict actual stability or meet regulations',
    )
    return linearize_transverse_system(
        M, D, h, mass=rigid.mass, beam=hull.state.beam,
        tolerance_per_second=tolerance_per_second, assumptions=assumptions)


def from_resolved_planing_boat(boat, *, step_rad: float = 1e-5,
                               tolerance_per_second: float = 1e-8) -> LinearTransverseSystem:
    """Non-mutating facade; requires previously solved geometry and bound mass.

    Never invokes get_steady_trim()/get_forces() or changes PlaningBoat fields.
    Caller must verify that the precomputed state is at actual equilibrium.
    """
    snapshot = boat.get_mass_properties_snapshot()  # checks stale binding
    rigid = RigidInertia.from_snapshot(snapshot)
    hs = coefficient_state_from_planing_boat(boat)
    rs = from_planing_boat(boat)
    hd = evaluate_hull_derivatives(hs, source='lew2004_table63')
    rd = RollForceModel(rs, source_variant='lew1996_eq27').derivatives(step_rad=step_rad)
    return assemble_lewandowski_2004(
        hd, rd, rigid, roll_state=rs,
        tolerance_per_second=tolerance_per_second)
