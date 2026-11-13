"""Rigid coordinate transforms for OpenPlaning mass properties.

A transform maps source-frame coordinates into target-frame coordinates:

    r_target = R @ r_source + t

Here R is a proper orthogonal matrix (det R = +1), and t is the
**source-frame origin expressed in target coordinates**, both in metres.
An intrinsic mass inertia tensor about the component CG transforms as
I_target = R @ I_source @ R.T. Translation affects the CG but *not*
the intrinsic inertia about that CG.  The parallel-axis theorem is
applied separately when an inertia about a different point is requested.

This module does not assume the axes or moment signs of Lewandowski.
The optional ``FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN`` conversion
changes between explicitly named right-handed coordinate systems; it
is not a claim that a specific publication uses either system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec3(value, label: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite three-component vector")
    return array


def _matrix3(value, label: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return array


def _rotation3(value) -> np.ndarray:
    R = _matrix3(value, "rotation")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-10, rtol=1e-10):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("rotation must preserve orientation (determinant +1)")
    return R


@dataclass(frozen=True, eq=False)
class RigidTransform:
    """Map coordinates from *source* to *target*: ``target = R @ source + t``.

    ``translation`` is the source origin, expressed in target coordinates.
    This class handles position, direction and intrinsic inertia transforms;
    products of inertia are encoded as NEGATIVE off-diagonal tensor entries.
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _rotation3(self.rotation))
        object.__setattr__(self, "translation", _vec3(self.translation, "translation"))

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls()

    @classmethod
    def from_axis_angle(cls, axis, angle_rad: float, *, translation=(0.0, 0.0, 0.0)) -> "RigidTransform":
        """Active right-handed rotation (Rodrigues); angle in radians."""
        axis = _vec3(axis, "axis")
        norm = float(np.linalg.norm(axis))
        angle = float(angle_rad)
        if norm == 0 or not np.isfinite(angle):
            raise ValueError("axis must be nonzero and angle_rad finite")
        u = axis / norm
        ux, uy, uz = u
        K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        s, c = np.sin(angle), np.cos(angle)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        return cls(R, translation)

    def point(self, position) -> np.ndarray:
        return self.rotation @ _vec3(position, "position") + self.translation

    def direction(self, vector) -> np.ndarray:
        return self.rotation @ _vec3(vector, "direction")

    def inertia_cg(self, tensor) -> np.ndarray:
        """Rotate physical inertia about its CG (kg m²); no translation."""
        I = _matrix3(tensor, "inertia tensor")
        if not np.allclose(I, I.T, rtol=1e-12, atol=1e-10):
            raise ValueError("inertia tensor must be symmetric")
        out = self.rotation @ I @ self.rotation.T
        return (out + out.T) / 2.0

    def inverse(self) -> "RigidTransform":
        R = self.rotation.T
        return RigidTransform(R, -R @ self.translation)

    def compose(self, before: "RigidTransform") -> "RigidTransform":
        """Return ``self ∘ before``; applies *before*, then *self*."""
        if not isinstance(before, RigidTransform):
            raise TypeError("before must be a RigidTransform")
        return RigidTransform(
            self.rotation @ before.rotation,
            self.rotation @ before.translation + self.translation,
        )


# Both coordinate systems are right-handed, related by 180 degrees about x.
# x forward in both; y port / z up becomes y starboard / z down.
FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN = RigidTransform(
    np.diag([1.0, -1.0, -1.0]), np.zeros(3)
)
