"""28C.4: explicit spatial-frame and descriptor-coordinate audit utilities.

This module does NOT fix/alter Lewandowski coefficients or the existing 28C.3
assembly. Source body convention is x forward, y starboard, z down (FSD);
OpenPlaning mass convention x forward, y port, z up (FPU). These are related
by a proper 180-degree rotation about x: R = diag(1,-1,-1).

Convention for inertia tensor: I_xz_tensor = - integral(x*z dm).
The book's inertial equation K=Ixx*p_dot-P_xz*r_dot uses the raw P_xz.
Thus P_xz(book FSD) = I_xz_tensor(OpenPlaning FPU), not its negative.
This is a *frame-interpretation audit*, NOT evidence that every hydrodynamic
coefficient of the 28C.3 assembler already uses the book FSD frame.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
import numpy as np

R_FPU_TO_FSD = np.diag([1., -1., -1.])
# Ordering of reduced descriptor rates q=[v, p=phi_dot, r].
# For 180 degrees about x, linear y and angular z change sign, roll x does not.
T_Q_FPU_TO_FSD = np.diag([-1., 1., -1.])
# Force/moment rows [Y, K, N] transform in exactly this manner.
T_W_FPU_TO_FSD = np.diag([-1., 1., -1.])
# x=[v, phi, p, r]. Both phi and p are roll about x.
T_X_FPU_TO_FSD = np.diag([-1., 1., 1., -1.])


def _valid_rotation(matrix) -> np.ndarray:
    R = np.asarray(matrix, dtype=float)
    if R.shape != (3,3) or not np.all(np.isfinite(R)):
        raise ValueError('rotation must be finite 3x3')
    if not np.allclose(R.T @ R, np.eye(3), rtol=1e-12, atol=1e-12):
        raise ValueError('rotation must be orthonormal')
    if not np.isclose(np.linalg.det(R), 1, rtol=1e-12, atol=1e-12):
        raise ValueError('rotation must be proper, det=+1')
    return R.copy()


def transform_spatial_quantities(rotation, *, vector=None, moment=None, inertia=None):
    """Transform polar force/velocity, axial moment, and physical inertia.

    Only proper rotations; no translational moment shifting is performed.
    Returns a dictionary with the quantities that were provided.
    """
    R = _valid_rotation(rotation)
    result = {}
    for key, value in (('vector', vector), ('moment', moment)):
        if value is not None:
            x = np.asarray(value, dtype=float)
            if x.shape != (3,) or not np.all(np.isfinite(x)):
                raise ValueError(key + ' must be finite three-vector')
            result[key] = R @ x
    if inertia is not None:
        I = np.asarray(inertia, dtype=float)
        if I.shape != (3,3) or not np.all(np.isfinite(I)) or not np.allclose(I,I.T):
            raise ValueError('inertia must be symmetric finite 3x3')
        result['inertia'] = R @ I @ R.T
    return result


def book_xz_raw_from_fpu_tensor(inertia_fpu) -> float:
    """Raw P_xz = integral(x_FSD*z_FSD dm) for CG-centered input tensor.

    The FPU->FSD rotation flips z while preserving x. The input is a full
    tensor about the center of gravity in FPU axes; output kg*m^2.
    """
    Ibook = transform_spatial_quantities(R_FPU_TO_FSD, inertia=inertia_fpu)['inertia']
    return float(-Ibook[0,2])


def fpu_xz_tensor_from_book_raw(raw_product: float) -> float:
    """Inverse of the above map; this returns the FPU tensor entry."""
    x = float(raw_product)
    if not isfinite(x):
        raise ValueError('raw product must be finite')
    return x


@dataclass(frozen=True, eq=False)
class DescriptorTransform:
    """Reduced 3-DOF coordinate transform q_B=Tq q_A, w_B=Tw w_A,
    phi_B=s phi_A, with the kinematic compatibility p_B=s p_A.

    A general rotation of x,y,z is NOT necessarily representable here:
    rotating x into y or z mixes retained with omitted DOFs. We permit only
    explicit diagonal sign/scaling transforms of the reduced coordinates.
    For physical FPU->FSD, use `physical_fpu_to_fsd()`.
    """
    q: np.ndarray
    wrench: np.ndarray
    heel_scale: float

    def __post_init__(self):
        q=np.asarray(self.q,dtype=float)
        w=np.asarray(self.wrench,dtype=float)
        h=float(self.heel_scale)
        if q.shape!=(3,3) or w.shape!=(3,3) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(w)):
            raise ValueError('coordinate matrices must be finite 3x3')
        if not isfinite(h) or h == 0:
            raise ValueError('heel scale must be finite and nonzero')
        if np.linalg.matrix_rank(q)<3 or np.linalg.matrix_rank(w)<3:
            raise ValueError('coordinate transforms must be invertible')
        # Do not allow arbitrary rate permutations that alter the kinematic definition p=dphi/dt.
        if not np.allclose(q[1,:], [0,h,0], atol=1e-12, rtol=1e-12):
            raise ValueError('p coordinate must transform identically to phi')
        object.__setattr__(self,'q',q.copy())
        object.__setattr__(self,'wrench',w.copy())
        object.__setattr__(self,'heel_scale',h)

    @classmethod
    def physical_fpu_to_fsd(cls) -> 'DescriptorTransform':
        return cls(T_Q_FPU_TO_FSD, T_W_FPU_TO_FSD, 1.)

    @property
    def state(self) -> np.ndarray:
        T=np.zeros((4,4));T[0,0]=self.q[0,0];T[1,1]=self.heel_scale
        T[2,0]=self.q[1,0];T[2,2]=self.q[1,1];T[2,3]=self.q[1,2]
        T[3,0]=self.q[2,0];T[3,2]=self.q[2,1];T[3,3]=self.q[2,2]
        # For the physical rotations and diagonal coordinate scalings used
        # in this audit, q[0,1], q[0,2] are zero. Support more general q
        # by completing the first and last rows from q explicitly.
        T[0,2]=self.q[0,1];T[0,3]=self.q[0,2]
        return T

    def descriptor(self, M, D, h):
        """Transform M*qdot + D*q + h*phi = 0, preserving solutions."""
        M=np.asarray(M,float);D=np.asarray(D,float);h=np.asarray(h,float)
        if M.shape!=(3,3) or D.shape!=(3,3) or h.shape!=(3,) or not all(np.all(np.isfinite(a)) for a in (M,D,h)):
            raise ValueError('M,D,h must be finite with shapes (3,3),(3,3),(3,)')
        invq=np.linalg.inv(self.q)
        return self.wrench@M@invq,self.wrench@D@invq,self.wrench@h/self.heel_scale

    def derivative_blocks(self, added_acceleration, rate, heel):
        """Transform derivative matrices by the same chain rule as descriptor."""
        return self.descriptor(added_acceleration,rate,heel)

    def state_similarity(self, A):
        A=np.asarray(A,float)
        if A.shape!=(4,4) or not np.all(np.isfinite(A)):
            raise ValueError('A must be finite 4x4')
        S=self.state
        return S@A@np.linalg.inv(S)
