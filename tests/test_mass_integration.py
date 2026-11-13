"""Tests of explicit PlaningBoat integration; no Savitsky solver is invoked."""
import unittest
from types import SimpleNamespace

import numpy as np

from openplaning.frames import RigidTransform
from openplaning.mass_properties import MassProperties
from openplaning.mass_integration import (
    check_mass_consistency, bind_mass_properties,
    get_mass_properties_snapshot,
)


class MassIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.mass = MassProperties()
        self.mass.add_box(120.0, [5., 0., 1.2], [2., 3., 4.])
        result = self.mass.calculate()
        self.boat = SimpleNamespace(
            g=9.81, weight=120*9.81, lcg=5., vcg=1.2,
            r_g=np.sqrt(result.inertia_tensor[1, 1] / result.mass),
            tau=3., z_wl=.5,
        )

    def test_check_is_read_only(self):
        before = dict(self.boat.__dict__)
        report = check_mass_consistency(self.boat, self.mass)
        self.assertTrue(report.consistent)
        self.assertEqual(before, self.boat.__dict__)

    def test_bind_matches_and_does_not_modify_solver_inputs(self):
        before = dict(self.boat.__dict__)
        report = bind_mass_properties(self.boat, self.mass)
        self.assertTrue(report.consistent)
        for key, value in before.items():
            self.assertEqual(getattr(self.boat, key), value)
        snap = get_mass_properties_snapshot(self.boat)
        self.assertEqual(snap.mass, 120.)
        self.assertAlmostEqual(snap.inertia_terms["Iyy"], 200.)

    def test_reject_mismatch_without_any_assignment(self):
        self.boat.weight *= 1.1
        with self.assertRaisesRegex(ValueError, "weight"):
            bind_mass_properties(self.boat, self.mass)
        self.assertFalse(hasattr(self.boat, "_rigid_body_mass_snapshot"))

    def test_reject_lateral_cg(self):
        altered = MassProperties()
        altered.add_box(120., [5., .1, 1.2], [2., 3., 4.])
        report = check_mass_consistency(self.boat, altered)
        self.assertFalse(report.consistent)
        self.assertFalse(report.lateral_cg_ok)
        with self.assertRaisesRegex(ValueError, "lateral CG"):
            bind_mass_properties(self.boat, altered)

    def test_rotated_source_model_then_bind(self):
        # Source origin 10 m farther aft in x; translate source -> boat.
        source = MassProperties()
        source.add_box(120., [-5., 0., 1.2], [2., 3., 4.])
        T = RigidTransform(translation=[10., 0., 0.])
        self.assertTrue(check_mass_consistency(
            self.boat, source, frame_transform=T).consistent)
        bind_mass_properties(self.boat, source, frame_transform=T)
        np.testing.assert_allclose(
            get_mass_properties_snapshot(self.boat).center_of_gravity, [5, 0, 1.2])

    def test_bind_is_snapshot_not_live_reference(self):
        bind_mass_properties(self.boat, self.mass)
        self.mass.add_point_mass(5, [100, 0, 0])
        snap = get_mass_properties_snapshot(self.boat)
        self.assertEqual(snap.mass, 120)
        np.testing.assert_allclose(snap.center_of_gravity, [5, 0, 1.2])
        self.assertFalse(snap.inertia_tensor.flags.writeable)

    def test_stale_snapshot_rejected_after_boat_change(self):
        bind_mass_properties(self.boat, self.mass)
        self.boat.r_g *= 1.01
        with self.assertRaisesRegex(RuntimeError, "stale"):
            get_mass_properties_snapshot(self.boat)

    def test_rebinding_failure_preserves_old_snapshot(self):
        bind_mass_properties(self.boat, self.mass)
        old = get_mass_properties_snapshot(self.boat)
        self.boat.lcg = 8.
        with self.assertRaises(ValueError):
            bind_mass_properties(self.boat, self.mass)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            get_mass_properties_snapshot(self.boat)
        self.boat.lcg = 5.
        self.assertEqual(get_mass_properties_snapshot(self.boat).mass, old.mass)

    def test_inertia_product_signs(self):
        mass = MassProperties()
        mass.add_point_mass(1., [1, 0, 1])
        mass.add_point_mass(1., [-1, 0, -1])
        result = mass.calculate()
        boat = SimpleNamespace(
            g=9.81, weight=19.62, lcg=0, vcg=0,
            r_g=np.sqrt(result.inertia_tensor[1, 1] / result.mass),
        )
        bind_mass_properties(boat, mass)
        terms = get_mass_properties_snapshot(boat).inertia_terms
        self.assertEqual(terms["Ixz_tensor"], -2)
        self.assertEqual(terms["Pxz_raw"], 2)

    def test_tolerance_validation(self):
        with self.assertRaises(ValueError):
            check_mass_consistency(self.boat, self.mass, rtol=-1)
        with self.assertRaises(ValueError):
            check_mass_consistency(self.boat, self.mass, atol=np.inf)

    def test_no_snapshot_raises(self):
        with self.assertRaises(RuntimeError):
            get_mass_properties_snapshot(self.boat)

    def test_only_massresult_accepted(self):
        with self.assertRaises(TypeError):
            bind_mass_properties(self.boat, [1, 2, 3])

    def test_nonfinite_current_boat_input(self):
        self.boat.vcg = float('nan')
        with self.assertRaises(ValueError):
            check_mass_consistency(self.boat, self.mass)


if __name__ == '__main__':
    unittest.main()
