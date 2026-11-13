"""Opt-in link between PlaningBoat and rigid-body mass properties.

No equilibrium is solved or existing boat inputs silently modified.
Coordinate contract: x forward from stern, y to port, z up from keel datum.
The mass tensor uses negative off-diagonals: I_xz=-int(x*z dm).
The reported *raw* xz product is -I_xz, and is NOT assigned to a
Lewandowski equation until its sign/axis convention is verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .frames import RigidTransform
from .mass_properties import MassProperties, MassResult


@dataclass(frozen=True)
class MassConsistencyReport:
    """Comparison of existing inputs with the component-derived values.

    ``differences`` maps name to (boat_value, mass_derived_value).
    A nonzero lateral center of gravity is incompatible with the symmetric
    PlaningBoat centerline assumption. No solver mutation occurs.
    """

    expected_inputs: dict[str, float]
    differences: dict[str, tuple[float, float]]
    lateral_cg: float
    lateral_cg_ok: bool

    @property
    def consistent(self) -> bool:
        return not self.differences and self.lateral_cg_ok


@dataclass(frozen=True)
class RigidBodyMassSnapshot:
    """Calculated mass/inertia in PlaningBoat body axes at bind time."""

    mass: float
    center_of_gravity: np.ndarray
    inertia_tensor: np.ndarray
    rtol: float = 1e-6
    atol: float = 1e-8
    lateral_tolerance: float = 1e-6

    @classmethod
    def from_result(
        cls, result: MassResult, *, rtol: float = 1e-6,
        atol: float = 1e-8, lateral_tolerance: float = 1e-6,
    ) -> "RigidBodyMassSnapshot":
        center = np.array(result.center_of_gravity, dtype=float, copy=True)
        tensor = np.array(result.inertia_tensor, dtype=float, copy=True)
        center.setflags(write=False)
        tensor.setflags(write=False)
        return cls(float(result.mass), center, tensor, rtol, atol, lateral_tolerance)

    @property
    def inertia_terms(self) -> dict[str, float]:
        """Physical inertia tensor components, not added mass (kg m²)."""
        I = self.inertia_tensor
        return {
            "Ixx": float(I[0, 0]),
            "Iyy": float(I[1, 1]),
            "Izz": float(I[2, 2]),
            "Ixy_tensor": float(I[0, 1]),
            "Ixz_tensor": float(I[0, 2]),
            "Iyz_tensor": float(I[1, 2]),
            "Pxz_raw": float(-I[0, 2]),
        }


def _resolve_result(
    model: Union[MassProperties, MassResult],
    frame_transform: Optional[RigidTransform],
) -> MassResult:
    if isinstance(model, MassProperties):
        result = model.calculate()
    elif isinstance(model, MassResult):
        result = model
    else:
        raise TypeError("mass properties must be a MassProperties or MassResult")
    if frame_transform is not None:
        if not isinstance(frame_transform, RigidTransform):
            raise TypeError("frame_transform must be RigidTransform (source -> PlaningBoat frame)")
        result = result.transformed(frame_transform)
    return result


def _scalar(value, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite real scalar") from exc
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def check_mass_consistency(
    boat,
    mass_properties: Union[MassProperties, MassResult],
    *,
    frame_transform: Optional[RigidTransform] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    lateral_tolerance: float = 1e-6,
) -> MassConsistencyReport:
    """Compare model against current PlaningBoat inputs without mutation.

    ``frame_transform`` maps model coordinates to the established boat frame.
    No automatic inference from CAD axes is performed. All scalar checks
    precede constructing a report. Lateral tolerance is an absolute CG
    distance in metres, independent of the input agreement tolerances.
    """
    rtol = _scalar(rtol, "rtol")
    atol = _scalar(atol, "atol")
    lateral_tolerance = _scalar(lateral_tolerance, "lateral_tolerance")
    if min(rtol, atol, lateral_tolerance) < 0:
        raise ValueError("tolerances must be nonnegative")
    result = _resolve_result(mass_properties, frame_transform)
    M = _scalar(result.mass, "calculated mass", positive=True)
    x, y, z = np.asarray(result.center_of_gravity, dtype=float)
    I = np.asarray(result.inertia_tensor, dtype=float)
    if I.shape != (3, 3) or not np.all(np.isfinite(I)) or I[1, 1] < 0:
        raise ValueError("invalid physical inertia tensor")
    if not np.all(np.isfinite([x, y, z])):
        raise ValueError("invalid center of gravity")

    expected = {
        "weight": M * _scalar(boat.g, "boat.g", positive=True),
        "lcg": float(x),
        "vcg": float(z),
        "r_g": float(np.sqrt(I[1, 1] / M)),
    }
    actual = {k: _scalar(getattr(boat, k), f"boat.{k}") for k in expected}
    differences = {
        k: (actual[k], expected[k])
        for k in expected
        if not np.isclose(actual[k], expected[k], rtol=rtol, atol=atol)
    }
    return MassConsistencyReport(
        expected_inputs=expected,
        differences=differences,
        lateral_cg=float(y),
        lateral_cg_ok=bool(abs(y) <= lateral_tolerance),
    )


def bind_mass_properties(
    boat,
    mass_properties: Union[MassProperties, MassResult],
    *,
    frame_transform: Optional[RigidTransform] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    lateral_tolerance: float = 1e-6,
) -> MassConsistencyReport:
    """Attach an *immutable snapshot* only when all existing inputs match.

    No boat mass, CG, trim, geometry, solver cache, or forces are changed.
    Rebinding replaces the previous snapshot only after all checks succeed.
    An asymmetric loading state must first be resolved in an appropriate
    asymmetric dynamics/equilibrium model; it cannot be silently accepted.
    """
    result = _resolve_result(mass_properties, frame_transform)
    report = check_mass_consistency(
        boat, result, rtol=rtol, atol=atol,
        lateral_tolerance=lateral_tolerance,
    )
    if not report.consistent:
        parts = [
            f"{name}: boat={value:.9g}, derived={expected:.9g}"
            for name, (value, expected) in report.differences.items()
        ]
        if not report.lateral_cg_ok:
            parts.append(f"lateral CG y={report.lateral_cg:.9g} m is not zero")
        raise ValueError("Mass properties inconsistent with PlaningBoat: " + "; ".join(parts))
    # Assignment only after successful validations, avoiding partial updates.
    boat._rigid_body_mass_snapshot = RigidBodyMassSnapshot.from_result(
        result, rtol=rtol, atol=atol, lateral_tolerance=lateral_tolerance
    )
    return report


def get_mass_properties_snapshot(boat) -> RigidBodyMassSnapshot:
    """Return a defensive copy of the associated rigid-body mass snapshot."""
    snap = getattr(boat, "_rigid_body_mass_snapshot", None)
    if snap is None:
        raise RuntimeError("No mass properties bound; call boat.set_mass_properties(...)")
    report = check_mass_consistency(
        boat, MassResult(snap.mass, snap.center_of_gravity, snap.inertia_tensor),
        rtol=snap.rtol, atol=snap.atol, lateral_tolerance=snap.lateral_tolerance,
    )
    if not report.consistent:
        raise RuntimeError(
            "Bound mass properties are stale: PlaningBoat inputs changed; "
            "reconcile and rebind after updating the operating model"
        )
    return RigidBodyMassSnapshot.from_result(
        MassResult(snap.mass, snap.center_of_gravity, snap.inertia_tensor),
        rtol=snap.rtol, atol=snap.atol, lateral_tolerance=snap.lateral_tolerance,
    )
