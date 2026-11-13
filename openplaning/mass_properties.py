"""Rigid-body mass properties for OpenPlaning.

Coordinate convention (SI): x forward from the transom, y to port, z upward
from keel reference; right-handed body coordinates. Positions are metres.

Inertia tensors are mass moments of inertia (kg m²), with off-diagonal
entries I_xy = -integral(x*y dm), etc. All component tensors describe mass
about each component's own center of gravity, in body axes.

This module is independent of Savitsky equilibrium and added hydrodynamic mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .frames import RigidTransform


def _vector(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain three finite coordinates")
    return result.copy()


def _tensor(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(result, result.T, rtol=1e-12, atol=1e-10):
        raise ValueError(f"{label} must be symmetric")
    result = (result + result.T) / 2
    scale = max(float(np.max(np.abs(result))), 1.0)
    if np.linalg.eigvalsh(result)[0] < -1e-10 * scale:
        raise ValueError(f"{label} must be positive semidefinite")
    # Any physical 3D inertia tensor must satisfy the triangle inequalities
    # for its principal moments (e.g., Ix + Iy >= Iz).
    eig = np.linalg.eigvalsh(result)
    if eig[2] > eig[0] + eig[1] + 1e-10 * scale:
        raise ValueError(f"{label} violates principal inertia triangle inequality")
    return result.copy()


def _rotation(matrix) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-10, rtol=1e-10):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=1e-10):
        raise ValueError("rotation must be proper (determinant +1)")
    return result.copy()


@dataclass(frozen=True, eq=False)
class MassComponent:
    """A rigid component, with tensor about its own CG in boat coordinates."""

    mass: float
    center: np.ndarray
    inertia_cg: np.ndarray
    label: str = ""

    def __post_init__(self):
        m = float(self.mass)
        if not np.isfinite(m) or m <= 0:
            raise ValueError("component mass must be strictly positive and finite")
        object.__setattr__(self, "mass", m)
        object.__setattr__(self, "center", _vector(self.center, "center"))
        object.__setattr__(self, "inertia_cg", _tensor(self.inertia_cg, "inertia_cg"))


@dataclass(frozen=True, eq=False)
class MassResult:
    mass: float
    center_of_gravity: np.ndarray
    inertia_tensor: np.ndarray

    @property
    def radii_of_gyration(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.inertia_tensor), 0.0) / self.mass)

    @property
    def principal_moments(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.inertia_tensor)

    def transformed(self, transform: RigidTransform) -> "MassResult":
        """Express the same rigid body in the transform's target frame.

        Inertia remains about the COM; translation changes COM only.
        """
        if not isinstance(transform, RigidTransform):
            raise TypeError("transform must be a RigidTransform")
        return MassResult(
            mass=self.mass,
            center_of_gravity=transform.point(self.center_of_gravity),
            inertia_tensor=transform.inertia_cg(self.inertia_tensor),
        )

    def inertia_at(self, point: Iterable[float]) -> np.ndarray:
        """Mass inertia about a parallel set of axes through `point` (m)."""
        d = self.center_of_gravity - _vector(point, "point")
        return self.inertia_tensor + self.mass * ((d @ d) * np.eye(3) - np.outer(d, d))


class MassProperties:
    """Collection of rigid components; no hydrodynamic added mass included."""

    def __init__(self, components: Iterable[MassComponent] = ()):
        self._components: list[MassComponent] = []
        for component in components:
            self.add_component(component)

    @property
    def components(self) -> tuple[MassComponent, ...]:
        return tuple(self._components)

    def add_component(self, component: MassComponent) -> MassComponent:
        if not isinstance(component, MassComponent):
            raise TypeError("component must be a MassComponent")
        self._components.append(component)
        return component

    def add_point_mass(self, mass: float, position, *, label: str = "") -> MassComponent:
        return self.add_component(MassComponent(mass, position, np.zeros((3, 3)), label))

    def add_box(self, mass: float, center, dimensions, *, rotation=None, label: str = "") -> MassComponent:
        """Solid uniform rectangular prism. Dimensions along local x,y,z (m).

        `rotation` maps local coordinates to body coordinates; default identity.
        """
        a, b, c = _vector(dimensions, "dimensions")
        if min(a, b, c) <= 0:
            raise ValueError("box dimensions must be strictly positive")
        m = float(mass)
        if not np.isfinite(m) or m <= 0:
            raise ValueError("box mass must be positive and finite")
        local = (m / 12.0) * np.diag([b*b+c*c, a*a+c*c, a*a+b*b])
        rot = np.eye(3) if rotation is None else _rotation(rotation)
        return self.add_component(MassComponent(m, center, rot @ local @ rot.T, label))

    def transformed(self, transform: RigidTransform) -> "MassProperties":
        """Return a NEW component collection expressed in the target frame.

        Component CG positions and intrinsic tensors are transformed.
        The source collection is not modified.
        """
        if not isinstance(transform, RigidTransform):
            raise TypeError("transform must be a RigidTransform")
        return MassProperties(
            MassComponent(
                component.mass,
                transform.point(component.center),
                transform.inertia_cg(component.inertia_cg),
                component.label,
            )
            for component in self._components
        )

    def calculate(self) -> MassResult:
        if not self._components:
            raise ValueError("at least one mass component is required")
        m = sum(c.mass for c in self._components)
        cg = sum((c.mass * c.center for c in self._components), np.zeros(3)) / m
        tensor = np.zeros((3, 3))
        for c in self._components:
            d = c.center - cg
            tensor += c.inertia_cg + c.mass * ((d @ d) * np.eye(3) - np.outer(d, d))
        return MassResult(mass=m, center_of_gravity=cg, inertia_tensor=(tensor + tensor.T) / 2)

    def compare_to_planing_boat(self, boat, *, rtol=1e-6, atol=1e-8) -> dict[str, tuple[float, float, bool]]:
        """Non-mutating compatibility report against existing PlaningBoat inputs.

        Returns parameter -> (existing value, component-derived value, matches).
        Does not mutate boat, change the solver, or assume a particular keel
        cross-section reference. The caller must verify the reference frame.
        """
        result = self.calculate()
        x, _, z = result.center_of_gravity
        expected = {
            "weight": result.mass * float(boat.g),
            "lcg": float(x),
            "vcg": float(z),
            "r_g": float(np.sqrt(result.inertia_tensor[1, 1] / result.mass)),
        }
        return {
            key: (float(getattr(boat, key)), value,
                  bool(np.isclose(float(getattr(boat, key)), value, rtol=rtol, atol=atol)))
            for key, value in expected.items()
        }
