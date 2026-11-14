"""28C.6: eigenmode continuation for the opt-in FSD transverse model.

Tracks two *oscillatory* eigenvalue branches by complex eigenvalue proximity
and overlap of dimensionless right eigenvectors. Branch continuity is a
numerical diagnostic, not an experimental identification of physical modes.

State x=[v, phi, p, r], p=phi_dot. SI input. The fixed reference scales
x to [v/U, phi, p*(b/U), r*(b/U)] for overlap comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import isfinite, pi
from typing import Callable, Sequence

import numpy as np

from .transverse import LinearTransverseSystem, RigidInertia
from .frame_resolved import assemble_lewandowski_2004_fsd
from .derivatives import coefficient_state_from_planing_boat, evaluate_hull_derivatives
from .lewandowski import from_planing_boat, RollForceModel


@dataclass(frozen=True)
class ModalSweep:
    ixz_tensor_fpu: np.ndarray             # kg m², sorted input points
    eigenvalues: np.ndarray                # (n,2), positive-imaginary modes, 1/s
    dimensionless_shapes: np.ndarray       # (n,2,4), [v/U,phi,p*b/U,r*b/U]
    kinetic_fractions: np.ndarray          # (n,2,3), approximate sway/roll/yaw rate kinetic energy
    dominant_branch: np.ndarray            # (n,), index 0 or 1
    dominance_brackets: tuple[tuple[float, float, float], ...]  # (left,right,linear estimate)
    assignment_margin: np.ndarray          # (n,), relative gap in assignment costs; 1 at initial point
    eigenvector_condition: np.ndarray      # (n,), condition of full 4x4 eigenvector basis
    complex_separation: np.ndarray         # (n,), |lambda0-lambda1| in 1/s
    notes: tuple[str, ...]

    @property
    def spectral_abscissa(self) -> np.ndarray:
        return np.max(self.eigenvalues.real, axis=1)

    @property
    def frequencies_hz(self) -> np.ndarray:
        return self.eigenvalues.imag / (2*pi)

    @property
    def decay_rates_per_second(self) -> np.ndarray:
        return -self.eigenvalues.real

    @property
    def kinetic_labels(self) -> np.ndarray:
        names = np.array(['sway_rate', 'roll_rate', 'yaw_rate'])
        return names[np.argmax(self.kinetic_fractions, axis=2)]

    def as_rows(self) -> list[dict]:
        """Flatten the sweep into plain serializable rows for CSV exports."""
        rows=[]
        for i, xz in enumerate(self.ixz_tensor_fpu):
            for j in range(2):
                frac=self.kinetic_fractions[i,j]
                rows.append(dict(ixz_tensor_fpu_kgm2=float(xz), branch=j,
                    real_per_s=float(self.eigenvalues[i,j].real),
                    imag_per_s=float(self.eigenvalues[i,j].imag),
                    frequency_hz=float(self.frequencies_hz[i,j]),
                    sway_kinetic_fraction=float(frac[0]),
                    roll_kinetic_fraction=float(frac[1]),
                    yaw_kinetic_fraction=float(frac[2]),
                    kinetic_label=str(self.kinetic_labels[i,j]),
                    dominant=bool(self.dominant_branch[i]==j),
                    spectral_abscissa_per_s=float(self.spectral_abscissa[i]),
                    assignment_margin=float(self.assignment_margin[i]),
                    eigenvector_condition=float(self.eigenvector_condition[i]),
                    complex_branch_separation_per_s=float(self.complex_separation[i])))
        return rows


def _validate_points(values: Sequence[float]) -> np.ndarray:
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or len(a)<3 or not np.all(np.isfinite(a)) or not np.all(np.diff(a)>0):
        raise ValueError('ixz_values must be a finite, strictly increasing sequence with at least 3 points')
    return a.copy()


def _eigenmode_frame(A: np.ndarray, scales: np.ndarray, *, imag_tol: float):
    eig, vec=np.linalg.eig(A)
    positive=np.flatnonzero(eig.imag>imag_tol)
    if len(positive)!=2:
        raise ValueError('mode tracking needs two distinct underdamped conjugate pairs; '
                         'a real/exceptional root or numerical coalescence may be present')
    raw=vec[:,positive]
    dimless=scales[:,None]*raw
    norms=np.linalg.norm(dimless,axis=0)
    if np.any(norms<=0):
        raise ValueError('zero eigenvector in modal tracking')
    dimless=dimless/norms
    full_condition=float(np.linalg.cond(vec))
    return eig[positive], dimless, raw, full_condition


def _energy_shares(mode: np.ndarray, mass: float, ixx: float, izz: float) -> np.ndarray:
    # Kinetic surrogate: only diagonal *rigid* inertia and velocity amplitudes.
    # Not complete fluid-body energy: added mass, cross inertia and potential
    # roll energy deliberately excluded. Never use as a certification metric.
    v,p,r=mode[[0,2,3]]
    energy=np.array([mass*abs(v)**2, ixx*abs(p)**2, izz*abs(r)**2],dtype=float)
    if not np.isfinite(energy).all() or energy.sum()<=0:
        raise ValueError('kinetic fractions could not be evaluated')
    return energy/energy.sum()


def _continuity_cost(prev_val, new_val, prev_shape, new_shape, *, spectral_scale: float):
    eig_cost=abs(new_val-prev_val)/spectral_scale
    overlap=abs(np.vdot(prev_shape,new_shape))**2
    # The weights are a documented tracking heuristic, NOT physical parameters.
    return .5*eig_cost+.5*(1-overlap)


def track_oscillatory_modes(
    ixz_values: Sequence[float],
    system_factory: Callable[[float], LinearTransverseSystem],
    *, speed: float, beam: float,
    mass: float, ixx: float, izz: float,
    imag_tolerance_per_second: float=1e-8,
) -> ModalSweep:
    """Track two complex-conjugate mode pairs on an ordered Ixz sweep.

    A user-supplied factory must supply LinearTransverseSystem for every Ixz.
    Modes with positive imaginary part are used once per conjugate pair.
    Normalized right eigenvectors are phase-aligned across adjacent points.
    Continuation may become ambiguous near degeneracies or exceptional points:
    assignment_margin and eigenvector_condition are returned as diagnostics.

    Frequency is never used as a physical mode name. Participation is merely
    a kinetic proxy from diagonal rigid mass and inertia, not full modal energy.
    """
    pts=_validate_points(ixz_values)
    for name,val in [('speed',speed),('beam',beam),('mass',mass),('ixx',ixx),('izz',izz)]:
        if not isfinite(val) or val<=0:
            raise ValueError(f'{name} must be positive and finite')
    if not isfinite(imag_tolerance_per_second) or imag_tolerance_per_second<0:
        raise ValueError('imag_tolerance_per_second must be finite and nonnegative')
    scales=np.array([1/speed,1.,beam/speed,beam/speed])
    n=len(pts)
    lam=np.empty((n,2),dtype=complex)
    shape=np.empty((n,2,4),dtype=complex)
    shares=np.empty((n,2,3))
    margins=np.ones(n)
    conds=np.empty(n)
    for i,pt in enumerate(pts):
        out=system_factory(float(pt))
        if not isinstance(out,LinearTransverseSystem):
            raise TypeError('system_factory must return LinearTransverseSystem')
        eig,dimless,raw,cond=_eigenmode_frame(out.state_matrix,scales,
                                              imag_tol=imag_tolerance_per_second)
        conds[i]=cond
        if i==0:
            order=np.argsort(eig.imag)
        else:
            prev=lam[i-1]; prevshape=shape[i-1]
            spectral_scale=max(1.,float(np.max(abs(prev))),float(np.max(abs(eig))))
            scores=[]
            for permutation in permutations(range(2)):
                score=sum(_continuity_cost(prev[j],eig[k],prevshape[j],dimless[:,k],
                                           spectral_scale=spectral_scale)
                          for j,k in enumerate(permutation))
                scores.append((score,permutation))
            scores.sort(key=lambda item:item[0])
            best,order=scores[0]; second=scores[1][0]
            margins[i]=(second-best)/max(second,1e-14)
        for j,k in enumerate(order):
            lam[i,j]=eig[k]
            v=dimless[:,k].copy()
            if i>0:
                phase=np.vdot(shape[i-1,j],v)
                if abs(phase)>1e-14:
                    v*=np.conj(phase)/abs(phase)
            shape[i,j]=v
            shares[i,j]=_energy_shares(raw[:,k],mass,ixx,izz)
    dominant=np.argmax(lam.real,axis=1)
    diff=lam[:,0].real-lam[:,1].real
    crossings=[]
    for i in range(n-1):
        if np.sign(diff[i])!=np.sign(diff[i+1]) or diff[i]==0:
            if diff[i+1]==diff[i]:
                root=0.5*(pts[i]+pts[i+1])
            else:
                root=pts[i]-diff[i]*(pts[i+1]-pts[i])/(diff[i+1]-diff[i])
            crossings.append((float(pts[i]),float(pts[i+1]),float(root)))
    gaps=abs(lam[:,0]-lam[:,1])
    for arr in (pts,lam,shape,shares,dominant,margins,conds,gaps):
        arr.setflags(write=False)
    return ModalSweep(pts,lam,shape,shares,dominant,tuple(crossings),margins,
                      conds,gaps,(
         'Two positive-imaginary branches; numerical continuation uses eigenvalue proximity and scaled right-eigenvector overlap.',
         'Dominant branch maximizes Re(lambda); a switch concerns damping dominance, not proof of physical mode conversion.',
         'Kinetic fractions use diagonal rigid inertia only: sway m|v|², roll Ixx|p|², yaw Izz|r|².',
         'Dimensionless shape scales: [v/U,phi,p*b/U,r*b/U]. Labels depend on this chosen diagnostic.',
         'Real-part crossing brackets are interpolated, not exact roots or proof of exceptional/avoided crossings.',
         'Large eigenvector condition or small assignment margin signals uncertain branch identification.',
         'FSD model and 1996 roll projections remain physically unvalidated; no safety verdict.',
    ))


def sweep_ixz_fsd(boat, ixz_values: Sequence[float], *, step_rad: float=1e-5) -> ModalSweep:
    """Sweep hypothetical rigid-body Ixz while holding solved FSD hull fixed.

    Requires an already solved boat and mass snapshot, checks symmetry and
    physical admissibility of EACH full 3D inertia tensor. Does not mutate boat.
    Frame convention: argument Ixz is PHYSICAL tensor[0,2] in FPU axes.
    """
    pts=_validate_points(ixz_values)
    snap=boat.get_mass_properties_snapshot()
    rigid=RigidInertia.from_snapshot(snap)
    I=np.asarray(snap.inertia_tensor,dtype=float)
    for xz in pts:
        trial=I.copy()
        trial[0,2]=trial[2,0]=xz
        eig=np.linalg.eigvalsh(trial)
        scale=max(1.,np.max(abs(eig)))
        if eig[0]<-1e-10*scale or eig[2]>eig[0]+eig[1]+1e-10*scale:
            raise ValueError(f'Ixz={xz:g} kg*m² gives nonphysical rigid-body inertia tensor')
    factory, hs = _fsd_factory_from_boat(boat, step_rad=step_rad)
    return track_oscillatory_modes(pts,factory,speed=hs.speed,beam=hs.beam,
                                    mass=rigid.mass,ixx=rigid.Ixx,izz=rigid.Izz)


@dataclass(frozen=True)
class DominanceCrossing:
    """Numerical real-part crossing of continuously tracked mode branches."""
    ixz_tensor_fpu: float
    bracket: tuple[float, float]
    complex_gap_per_second: float
    iterations: int
    minimum_assignment_margin: float


def refine_dominance_crossings(
    sweep: ModalSweep,
    system_factory: Callable[[float], LinearTransverseSystem],
    *, speed: float, beam: float,
    tolerance_ixz_kgm2: float = 1e-7,
    max_iterations: int = 60,
) -> tuple[DominanceCrossing, ...]:
    """Bisection on Re(lambda_branch0 - lambda_branch1), not on frequency.

    Uses the tracked left-endpoint eigenpairs as continuation anchors for each
    trial midpoint. A small assignment margin or near-degenerate eigenvector
    basis invalidates a confident physical branch label; inspect diagnostics.
    This method does NOT identify exceptional points or avoided crossings.
    """
    if not isfinite(speed) or not isfinite(beam) or speed <= 0 or beam <= 0:
        raise ValueError('speed and beam must be positive and finite')
    if not isfinite(tolerance_ixz_kgm2) or tolerance_ixz_kgm2 <= 0:
        raise ValueError('tolerance_ixz_kgm2 must be positive and finite')
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    scales = np.array([1/speed,1.,beam/speed,beam/speed])
    found=[]
    for x0,x1,_ in sweep.dominance_brackets:
        idx=int(np.argmin(abs(sweep.ixz_tensor_fpu-x0)))
        left=float(x0); right=float(x1)
        reference_eig=sweep.eigenvalues[idx].copy()
        reference_vec=sweep.dimensionless_shapes[idx].copy()
        fleft=float((reference_eig[0]-reference_eig[1]).real)
        min_margin=1.
        if abs(fleft)<1e-12:
            found.append(DominanceCrossing(left,(left,left),
                        abs(reference_eig[0]-reference_eig[1]),0,min_margin))
            continue
        for k in range(max_iterations):
            mid=(left+right)/2
            data=system_factory(mid)
            if not isinstance(data,LinearTransverseSystem):
                raise TypeError('system_factory must return LinearTransverseSystem')
            eig,vec,_,_=_eigenmode_frame(data.state_matrix,scales,imag_tol=1e-8)
            scale=max(1.,max(abs(eig)),max(abs(reference_eig)))
            costs=[]
            for order in permutations(range(2)):
                costs.append((sum(_continuity_cost(reference_eig[j],eig[z],reference_vec[j],vec[:,z],
                                  spectral_scale=scale) for j,z in enumerate(order)),order))
            costs.sort(key=lambda z:z[0]);best,order=costs[0]
            margin=(costs[1][0]-best)/max(costs[1][0],1e-14)
            min_margin=min(min_margin,margin)
            chosen=eig[list(order)]
            fmid=float((chosen[0]-chosen[1]).real)
            if fleft*fmid<=0:
                right=mid
            else:
                left=mid
                fleft=fmid
                reference_eig=chosen.copy()
                reference_vec=vec[:,list(order)].T.copy()
            if right-left<=tolerance_ixz_kgm2:
                break
        root=(left+right)/2
        final=system_factory(root)
        ev,v,_,_=_eigenmode_frame(final.state_matrix,scales,imag_tol=1e-8)
        scale=max(1.,max(abs(ev)),max(abs(reference_eig)))
        order=min(permutations(range(2)),key=lambda p:sum(
            _continuity_cost(reference_eig[j],ev[z],reference_vec[j],v[:,z],spectral_scale=scale)
            for j,z in enumerate(p)))
        vals=ev[list(order)]
        found.append(DominanceCrossing(float(root),(float(left),float(right)),
                     float(abs(vals[0]-vals[1])),k+1,float(min_margin)))
    return tuple(found)



def _fsd_factory_from_boat(boat, *, step_rad: float):
    """Bind a fixed solved state once, with an opt-in inertia-tensor sweep."""
    rigid=RigidInertia.from_snapshot(boat.get_mass_properties_snapshot())
    hs=coefficient_state_from_planing_boat(boat)
    rs=from_planing_boat(boat)
    hull=evaluate_hull_derivatives(hs,source='lew2004_table63')
    roll=RollForceModel(rs,source_variant='lew1996_eq27').derivatives(step_rad=step_rad)
    def factory(xz):
        rigid_at=RigidInertia(rigid.mass,rigid.Ixx,rigid.Izz,xz,
                              rigid.Ixy_tensor,rigid.Iyz_tensor)
        return assemble_lewandowski_2004_fsd(hull,roll,rigid_at,roll_state=rs)
    return factory, hs


def analyze_ixz_fsd(boat, ixz_values: Sequence[float], *, step_rad: float=1e-5,
                    tolerance_ixz_kgm2: float=1e-7) -> tuple[ModalSweep, tuple[DominanceCrossing, ...]]:
    """One-call physically admissible FSD inertia sweep and crossing refinement.

    Returns (sweep, crossings). These are *model* results, not sea-trial data.
    """
    sweep=sweep_ixz_fsd(boat, ixz_values,step_rad=step_rad)
    factory,hs=_fsd_factory_from_boat(boat,step_rad=step_rad)
    crossings=refine_dominance_crossings(sweep,factory,speed=hs.speed,
                    beam=hs.beam,tolerance_ixz_kgm2=tolerance_ixz_kgm2)
    return sweep,crossings
