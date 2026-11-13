# Mass properties — implementation 28B.1 / 28B.2

## Defined physical conventions

- Body-fixed right-handed coordinates: **x forward from transom**, **y to port**, **z upward from keel datum**.
- All positions and component CGs are in **metres**; component masses are **kg**, moments of inertia **kg m²**.
- The inertia tensor is the standard physical tensor: diagonal `∫(r²-x²)dm`; off diagonal `-∫xy dm` and analogues.
- No hydrodynamic added mass or inertial coefficients of surrounding fluid are included.
- Component tensors are about individual component CGs, expressed in the boat body frame.
- Rectangle prism dimensions align with local x/y/z; optional proper orthonormal rotation maps local to boat frame.
- Coordinates must share the same fixed reference and axes; potential ambiguities between the source paper and OpenPlaning require explicit conversion upstream of these routines.

## API

`MassProperties.add_point_mass(mass, position, label="")`

`MassProperties.add_box(mass, center, dimensions, rotation=None, label="")`

`MassProperties.add_component(MassComponent(mass, center, inertia_cg, label=""))`

`MassProperties.calculate()` -> `MassResult` with `mass`, `center_of_gravity`, `inertia_tensor`, `radii_of_gyration`, `principal_moments` and `inertia_at(point)`.

`MassProperties.compare_to_planing_boat(boat)` is a **read-only** check of `weight`, `lcg`, `vcg` and `r_g` against the calculated values; it does not modify the boat, synchronize values, or invoke equilibrium solving.

## Limitation and roadmap

- `PlaningBoat` currently uses `r_g` for the pitch inertia. Its presence does not provide roll or yaw inertia.
- A **point mass** deliberately has zero intrinsic moment of inertia; representing a large hull or engine with one point mass can significantly underestimate Ixx, Iyy, Izz.
- The box component assumes uniform density and rigid body. Other solids, tank free surfaces and time-dependent loading are not implemented.
- Future 28B.3: rigid-frame transformations and interface for importers; 28B.4: adapter for PlaningBoat with controlled coherence checks and no silent overwrites.
- Future 28C: evaluate Lewandowski coupling and products of inertia using explicit sign conventions.
- No empirical uncertainty quantification or physical validation is claimed.

## Run tests

From the repository root (Python with numpy installed):

`python -m unittest discover -s tests -p 'test_mass_properties.py' -v`
