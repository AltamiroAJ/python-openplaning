"""28C.10: EXPLICIT, NON-VALIDATED Lewandowski source variants.

No default formulation, no safety certification. The 2004 variant follows the
PRINTED Table 6.2/6.3 restoring terms, whereas the 1996 variant follows the
full per-side projection under the 28C.5 FPU->FSD interpretation.

Both use the identical hull-rate coefficients from 2004 Table 6.3 and the
same CG-centered rigid inertia in FSD. This is a controlled comparison of
heel-column hypotheses, NOT a claim that the 1996 and 2004 closures are
physically interchangeable. SI throughout; empirical angular definitions
retain the units specified in their sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np

from .derivatives import HullDerivatives, coefficient_state_from_planing_boat, evaluate_hull_derivatives
from .lewandowski import LewandowskiState, RollForceDerivatives, RollForceModel, from_planing_boat
from .source_audit import audit_roll_sources
from .transverse import RigidInertia, LinearTransverseSystem, _compare_states, linearize_transverse_system
from .frame_resolved import assemble_lewandowski_2004_fsd


class EvidenceLevel(str, Enum):
    """Required disclosure: neither level means physically validated."""
    DOCUMENTAL_FIDELITY = 'documental_fidelity'
    DIAGNOSTIC = 'diagnostic'


class SourceFormulation(str, Enum):
    """There is deliberately no DEFAULT/AUTO member."""
    LEWANDOWSKI_2004_TABULATED = 'lewandowski_2004_tabulated'
    LEWANDOWSKI_1996_FULL_PROJECTION = 'lewandowski_1996_full_projection'


@dataclass(frozen=True)
class SourceResolvedSystem:
    """Immutable labeled result; linear eigenvalues are implementation diagnostics.

    `linear_system` is the unchanged 28C.3 LinearTransverseSystem type. The
    publication/closure label and evidence level are mandatory typed fields.
    """
    formulation: SourceFormulation
    evidence_level: EvidenceLevel
    linear_system: LinearTransverseSystem
    reference: str
    frame: str
    units: str
    limitations: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.formulation, SourceFormulation):
            raise TypeError('formulation must be SourceFormulation')
        if not isinstance(self.evidence_level, EvidenceLevel):
            raise TypeError('evidence_level must be EvidenceLevel')
        if not isinstance(self.linear_system, LinearTransverseSystem):
            raise TypeError('linear_system must be LinearTransverseSystem')
        if not self.limitations or not self.reference or not self.frame or not self.units:
            raise ValueError('source provenance/limitations are mandatory')

    @property
    def eigenvalues(self):
        return self.linear_system.eigenvalues

    @property
    def spectral_abscissa(self):
        return self.linear_system.spectral_abscissa

    @property
    def heel_column(self):
        return self.linear_system.heel_column

    @property
    def state_matrix(self):
        return self.linear_system.state_matrix


def _inputs(hull, roll, rigid, roll_state):
    if not isinstance(hull, HullDerivatives) or hull.source != 'lew2004_table63':
        raise ValueError('only 2004 Table 6.3 hull rate derivatives permitted')
    hull.require_dimensional_consistency()
    if not isinstance(roll, RollForceDerivatives) or roll.source_variant != 'lew1996_eq27':
        raise ValueError('expected published 1996 Eq. 27 side-force result')
    if not isinstance(rigid, RigidInertia) or not isinstance(roll_state, LewandowskiState):
        raise TypeError('rigid must be RigidInertia, roll_state must be LewandowskiState')
    _compare_states(hull, roll_state)


def _tabulated_heel_column(state: LewandowskiState, *, step_rad: float = 2e-5):
    """Printed 2004 Tables 6.2/6.3 IN THE BOOK FSD FRAME.

    N_phi is formed only from the computed Y_phi and the CG/CP arm, never
    from a separate yaw regression. No 1996 full-projection term 2*F0*cos(beta).
    """
    a = audit_roll_sources(state, step_rad=step_rad)
    y = a.y_phi_2004_table63_N_per_rad
    n = y * (state.lcp - state.lcg)
    k = a.k_phi_2004_table62_Nm_per_rad + a.k_phi_static_2004_Nm_per_rad
    h = np.array([y, k, n], dtype=float)
    if not np.all(np.isfinite(h)):
        raise ValueError('non-finite 2004 restoring coefficients')
    return h


def _base_matrices(hull: HullDerivatives, rigid: RigidInertia):
    """2004 6.8--6.12 FSD matrices, sharing 28C.5 rate and inertia conventions."""
    v = hull.values
    m, u = rigid.mass, hull.state.speed
    pxz = rigid.Ixz_tensor  # P_xz,FSD = tensor I_xz,FPU; audited 28C.4
    M = np.array([
        [v['Y_v_dot']-m, v['Y_phi_ddot'], v['Y_r_dot']],
        [v['K_v_dot'], v['K_phi_ddot']-rigid.Ixx, v['K_r_dot']+pxz],
        [v['N_v_dot'], v['N_phi_ddot']+pxz, v['N_r_dot']-rigid.Izz],
    ], dtype=float)
    D = np.array([
        [v['Y_v'], v['Y_phi_dot'], v['Y_r']-m*u],
        [v['K_v'], v['K_phi_dot'], v['K_r']],
        [v['N_v'], v['N_phi_dot'], v['N_r']],
    ], dtype=float)
    return M, D


def assemble_lewandowski_2004_tabulated(
    hull: HullDerivatives, roll: RollForceDerivatives, rigid: RigidInertia,
    *, roll_state: LewandowskiState, step_rad: float = 2e-5,
    tolerance_per_second: float = 1e-8,
) -> SourceResolvedSystem:
    """Documental-fidelity 2004 Table 6.2/6.3 assembly; NOT physically validated.

    Uses Y_phi=2*F_phi*sin(beta), N_phi=Y_phi*(LCP-LCG), and
    K_phi=K_Dphi+K_static printed in the tables. F_phi and K_static are
    taken independently from the source audit, not differentiated from the
    1996 full lateral projection. The input `roll` verifies source identity.
    """
    _inputs(hull, roll, rigid, roll_state)
    h = _tabulated_heel_column(roll_state, step_rad=step_rad)
    M,D = _base_matrices(hull, rigid)
    system = linearize_transverse_system(M,D,h,mass=rigid.mass,beam=hull.state.beam,
        tolerance_per_second=tolerance_per_second,
        assumptions=('2004 printed Tables 6.2/6.3; FSD body frame',
                     'Documental fidelity ONLY; no empirical physical validation',
                     '1996 full projection 2*F0*cos(beta) intentionally not included',
                     'No appendage, asymmetric hull, or 6DOF dynamics'))
    return SourceResolvedSystem(SourceFormulation.LEWANDOWSKI_2004_TABULATED,
        EvidenceLevel.DOCUMENTAL_FIDELITY, system,
        'Lewandowski (2004), Tables 6.2 and 6.3, pp. 369–370',
        'FSD: x forward, y starboard, z downward; reference CG',
        'SI: force N, moments N m, derivatives per radian',
        ('Formula transcription is not experimental validation',
         'Omission of the full-projection geometric term is not physically resolved',
         'Appendages and additional lateral static derivatives are excluded'))


def assemble_lewandowski_1996_full_projection(
    hull: HullDerivatives, roll: RollForceDerivatives, rigid: RigidInertia,
    *, roll_state: LewandowskiState, tolerance_per_second: float = 1e-8,
) -> SourceResolvedSystem:
    """Diagnostic 1996 full side-force projection / 28C.5 FSD hypothesis.

    Never claim this is an unmodified 1996 published complete EOM: hull-rate
    derivatives still come from the 2004 table, and the 1996 projection is
    transformed from FPU to FSD. This method reproduces the 28C.5 FSD model.
    """
    _inputs(hull,roll,rigid,roll_state)
    system = assemble_lewandowski_2004_fsd(hull,roll,rigid,roll_state=roll_state,
                                             tolerance_per_second=tolerance_per_second)
    return SourceResolvedSystem(SourceFormulation.LEWANDOWSKI_1996_FULL_PROJECTION,
        EvidenceLevel.DIAGNOSTIC,system,
        'Lewandowski (1996), Eq. 27 and signed side projection; combined with 2004 hull rates',
        'FSD: 1996 projection interpreted in FPU then transformed to FSD',
        'SI: force N, moments N m, derivatives per radian',
        ('Hybrid 1996 heel forces / 2004 rates; NOT an independent 1996 model',
         'Port/starboard projection physics not validated',
         'Appendages omitted'))


def from_resolved_planing_boat(
    boat, *, formulation: SourceFormulation, step_rad: float = 1e-5,
    tolerance_per_second: float = 1e-8,
) -> SourceResolvedSystem:
    """PUBLIC FACADE: formulation REQUIRED; absence raises TypeError.

    No silent default. Unlike the historical
    `openplaning.stability.transverse.from_resolved_planing_boat` function,
    this entry point requires a typed SourceFormulation, preventing accidental
    promotion of the newer 2004 tabulated implementation.
    """
    if not isinstance(formulation, SourceFormulation):
        raise TypeError('formulation must be an explicit SourceFormulation enum')
    rigid = RigidInertia.from_snapshot(boat.get_mass_properties_snapshot())
    hs = coefficient_state_from_planing_boat(boat)
    rs = from_planing_boat(boat)
    hd = evaluate_hull_derivatives(hs,source='lew2004_table63')
    rd = RollForceModel(rs,source_variant='lew1996_eq27').derivatives(step_rad=step_rad)
    if formulation == SourceFormulation.LEWANDOWSKI_2004_TABULATED:
        return assemble_lewandowski_2004_tabulated(hd,rd,rigid,roll_state=rs,
                  step_rad=step_rad,tolerance_per_second=tolerance_per_second)
    if formulation == SourceFormulation.LEWANDOWSKI_1996_FULL_PROJECTION:
        return assemble_lewandowski_1996_full_projection(hd,rd,rigid,roll_state=rs,
                  tolerance_per_second=tolerance_per_second)
    raise ValueError('unsupported source formulation')


def assert_28c7_reference(
    boat, *, relative_tolerance: float=2e-5, absolute_tolerance: float=0.1,
):
    """Integration acceptance for the REAL 1600 kg/12 m/s boat snapshot.

    User-observed 28C.7 benchmarks (do not manufacture missing boat inputs):
      Y_phi_2004 = -8867.644 N/rad,
      N_phi_2004 = -181.9909 Nm/rad,
      Y_phi_1996_full = +6950.227 N/rad.
    Requires the SAME actual resolved PlaningBoat and mass snapshot used for
    those numbers. Raises AssertionError on mismatch, no parameter tuning.
    """
    if not isfinite(relative_tolerance) or not isfinite(absolute_tolerance):
        raise ValueError('tolerances must be finite')
    if min(relative_tolerance,absolute_tolerance)<0:
        raise ValueError('tolerances must be nonnegative')
    state=from_planing_boat(boat)
    if not np.isclose(state.speed, 12., rtol=0, atol=1e-6):
        raise ValueError('R5 reference expects speed = 12 m/s')
    snapshot=boat.get_mass_properties_snapshot()
    if not np.isclose(float(snapshot.mass),1600.,rtol=0,atol=1e-3):
        raise ValueError('R5 reference expects mass = 1600 kg')
    audit=audit_roll_sources(state)
    observed = {
        'Y_phi_2004':audit.y_phi_2004_table63_N_per_rad,
        'N_phi_2004':audit.n_phi_2004_table63_Nm_per_rad,
        'Y_phi_1996_full':audit.y_phi_full_projection_N_per_rad,
    }
    target={'Y_phi_2004':-8867.644,'N_phi_2004':-181.9909,
            'Y_phi_1996_full':6950.227}
    for key in target:
        if not np.isclose(observed[key],target[key],rtol=relative_tolerance,
                          atol=absolute_tolerance):
            raise AssertionError(f'{key}: actual={observed[key]:.9f}, reference={target[key]:.9f}')
    # Critically: the NEW assembly must reproduce the independent 28C.7 values.
    tab=from_resolved_planing_boat(boat,
        formulation=SourceFormulation.LEWANDOWSKI_2004_TABULATED)
    full=from_resolved_planing_boat(boat,
        formulation=SourceFormulation.LEWANDOWSKI_1996_FULL_PROJECTION)
    for name,value,expected in (
        ('new_tabulated_Y_phi',tab.heel_column[0],target['Y_phi_2004']),
        ('new_tabulated_N_phi',tab.heel_column[2],target['N_phi_2004']),
        ('new_full_projection_FSD_Y_phi',full.heel_column[0],-target['Y_phi_1996_full']),
    ):
        if not np.isclose(value,expected,rtol=relative_tolerance,atol=absolute_tolerance):
            raise AssertionError(f'{name}: actual={value:.9f}, reference={expected:.9f}')
    return {'reference':target,'audit':observed,
            'tabulated_spectral_abscissa':tab.spectral_abscissa,
            'full_projection_spectral_abscissa':full.spectral_abscissa,
            'full_projection_eigenvalues':full.eigenvalues.copy(),
            'tabulated_eigenvalues':tab.eigenvalues.copy()}
