"""28C.5: explicit frame-resolved *hypothesis* for Lewandowski (2004).

This is an opt-in alternative to the unchanged 28C.3 diagnostic assembly.
The 2004 table coefficients are interpreted in the author's FSD frame,
while RollForceModel follows dynroll's FPU *signed projection* convention.
Those signed side-force projections remain an unvalidated physical hypothesis.

Body axes (origin CG): FPU x-forward/y-port/z-up (mass model) and
FSD x-forward/y-starboard/z-down (Lewandowski book).
P_xz,FSD = integral(x_FSD*z_FSD dm) = +I_xz,tensor,FPU.

Inputs and outputs are SI. Beta/tau retain degrees in empirical regressions.
No geometry, trim, hydrodynamic state, or mass snapshot is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .derivatives import (
    HullDerivatives, coefficient_state_from_planing_boat,
    evaluate_hull_derivatives,
)
from .frame_audit import DescriptorTransform
from .lewandowski import (
    RollForceDerivatives, RollForceModel, from_planing_boat,
)
from .transverse import (
    LinearTransverseSystem, RigidInertia, _compare_states,
    assemble_lewandowski_2004, linearize_transverse_system,
)


@dataclass(frozen=True)
class FrameResolvedComparison:
    """Compare unchanged legacy 28C.3 and opt-in frame-resolved hypotheses."""
    legacy: LinearTransverseSystem
    fsd: LinearTransverseSystem
    spectral_abscissa_difference: float  # fsd minus legacy [1/s]
    eigenvalue_maximum_assignment_difference: float  # [1/s]


def _eig_assignment_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum max absolute complex-root difference over four bijections."""
    from itertools import permutations
    return float(min(max(abs(a[i] - b[j]) for i, j in enumerate(perm))
                     for perm in permutations(range(4))))


def assemble_lewandowski_2004_fsd(
    hull: HullDerivatives,
    roll: RollForceDerivatives,
    rigid: RigidInertia,
    *,
    roll_state,
    tolerance_per_second: float = 1e-8,
) -> LinearTransverseSystem:
    """Opt-in source-frame assembly of Eqs. 6.8--6.12.

    Explicit source assignments:
    - HullDerivatives['lew2004_table63']: interpreted as published FSD
      [Y_starboard, K_forward, N_down], with rates [v_starboard,p,r_down].
    - RollForceDerivatives from 1996 Eq.27: signed *dynroll/FPU* projections.
      For fixed phi about the x-axis, only Y_phi and N_phi reverse sign.
    - RigidInertia: FPU physical inertia tensor at the CG. Convert the
      raw product P_xz to FSD with R=diag(1,-1,-1).

    This is a coordinate-consistent *interpretation*, not a validation of the
    1996 port/starboard force projection, appendage forces, or sea-trial data.
    """
    if not isinstance(hull, HullDerivatives) or not isinstance(roll, RollForceDerivatives):
        raise TypeError('hull and roll must be labeled derivative objects')
    if not isinstance(rigid, RigidInertia):
        raise TypeError('rigid must be a RigidInertia')
    if hull.source != 'lew2004_table63':
        raise ValueError('only the dimensionally consistent 2004 table is permitted')
    hull.require_dimensional_consistency()
    if roll.source_variant != 'lew1996_eq27':
        raise ValueError('only the published 1996 Eq.27 roll variant is permitted')
    _compare_states(hull, roll_state)

    s = hull.state
    v = hull.values
    mass, U = rigid.mass, s.speed

    # All entries are about the common CG and use FSD rates/wrenches.
    # From the audited FPU->FSD rotation: P_xz,FSD = I_xz,tensor,FPU.
    # Do not use the misleading legacy property rigid.Ixz_book_raw.
    pxz_fsd = float(rigid.Ixz_tensor)

    M = np.array([
        [v['Y_v_dot'] - mass, v['Y_phi_ddot'], v['Y_r_dot']],
        [v['K_v_dot'], v['K_phi_ddot'] - rigid.Ixx, v['K_r_dot'] + pxz_fsd],
        [v['N_v_dot'], v['N_phi_ddot'] + pxz_fsd, v['N_r_dot'] - rigid.Izz],
    ], dtype=float)
    D = np.array([
        [v['Y_v'], v['Y_phi_dot'], v['Y_r'] - mass * U],
        [v['K_v'], v['K_phi_dot'], v['K_r']],
        [v['N_v'], v['N_phi_dot'], v['N_r']],
    ], dtype=float)
    # RollForceModel explicitly exports the dynroll/FPU-signed projections.
    # Rotating FPU -> FSD is not equivalent to negating Ixz alone.
    T_w = DescriptorTransform.physical_fpu_to_fsd().wrench
    h_fpu = np.array([roll.Y_phi_dynamic, roll.K_phi_total,
                      roll.N_phi_dynamic], dtype=float)
    h_fsd = T_w @ h_fpu

    return linearize_transverse_system(
        M, D, h_fsd, mass=mass, beam=s.beam,
        tolerance_per_second=tolerance_per_second,
        assumptions=(
            '28C.5: explicit FSD 2004 hull-coefficient interpretation',
            '1996 signed roll-force projections interpreted as FPU and rotated into FSD',
            'P_xz(book,FSD)=+Ixz_tensor(FPU), physical tensor at center of gravity',
            'Coriolis Y_r - mass*U retained in published FSD rate equations',
            'No 1997 variant, appendage terms, asymmetrical hull or 6-DOF coupling',
            'Unvalidated roll projection/source convention; diagnostic eigenvalues only',
        ),
    )


def from_resolved_planing_boat_fsd(
    boat, *, step_rad: float = 1e-5,
    tolerance_per_second: float = 1e-8,
) -> LinearTransverseSystem:
    """Read a previously solved boat with bound mass; never solve or mutate."""
    rigid = RigidInertia.from_snapshot(boat.get_mass_properties_snapshot())
    hs = coefficient_state_from_planing_boat(boat)
    rs = from_planing_boat(boat)
    hd = evaluate_hull_derivatives(hs, source='lew2004_table63')
    rd = RollForceModel(rs, source_variant='lew1996_eq27').derivatives(step_rad=step_rad)
    return assemble_lewandowski_2004_fsd(
        hd, rd, rigid, roll_state=rs,
        tolerance_per_second=tolerance_per_second,
    )


def compare_legacy_and_fsd(
    boat, *, step_rad: float = 1e-5,
    tolerance_per_second: float = 1e-8,
) -> FrameResolvedComparison:
    """Reproduce both variants at exactly the same solved state and mass."""
    rigid = RigidInertia.from_snapshot(boat.get_mass_properties_snapshot())
    hs = coefficient_state_from_planing_boat(boat)
    rs = from_planing_boat(boat)
    hd = evaluate_hull_derivatives(hs, source='lew2004_table63')
    rd = RollForceModel(rs, source_variant='lew1996_eq27').derivatives(step_rad=step_rad)
    old = assemble_lewandowski_2004(
        hd, rd, rigid, roll_state=rs,
        tolerance_per_second=tolerance_per_second,
    )
    new = assemble_lewandowski_2004_fsd(
        hd, rd, rigid, roll_state=rs,
        tolerance_per_second=tolerance_per_second,
    )
    return FrameResolvedComparison(
        old, new, new.spectral_abscissa - old.spectral_abscissa,
        _eig_assignment_distance(old.eigenvalues, new.eigenvalues),
    )
