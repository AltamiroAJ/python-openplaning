"""Analytical tests for body-frame transforms and mass inertia invariants."""
import unittest
import numpy as np

from openplaning.frames import (
    RigidTransform, FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN,
)
from openplaning.mass_properties import MassProperties, MassComponent


class RigidTransformTests(unittest.TestCase):

    def test_identity(self):
        T = RigidTransform.identity()
        np.testing.assert_allclose(T.point([1, 2, 3]), [1, 2, 3])
        np.testing.assert_allclose(T.direction([1, 2, 3]), [1, 2, 3])

    def test_translation_does_not_change_direction_or_inertia(self):
        T = RigidTransform(translation=[10, -20, 2])
        np.testing.assert_allclose(T.point([1, 2, 3]), [11, -18, 5])
        np.testing.assert_allclose(T.direction([1, 2, 3]), [1, 2, 3])
        np.testing.assert_allclose(T.inertia_cg(np.diag([2, 3, 4])), np.diag([2, 3, 4]))

    def test_right_handed_quarter_turn(self):
        T = RigidTransform.from_axis_angle([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(T.direction([1, 0, 0]), [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(T.inertia_cg(np.diag([2, 3, 4])),
                                   np.diag([3, 2, 4]), atol=1e-14)

    def test_inverse_round_trip(self):
        T = RigidTransform.from_axis_angle([1, 3, -2], 0.61, translation=[8, -1, 0.5])
        point = np.array([3.4, -2.2, 7.1])
        np.testing.assert_allclose(T.inverse().point(T.point(point)), point, atol=1e-13)

    def test_composition_order(self):
        A = RigidTransform.from_axis_angle([0, 0, 1], 0.2, translation=[1, 2, 3])
        B = RigidTransform.from_axis_angle([0, 1, 0], 0.7, translation=[-2, 1, 0])
        p = np.array([3.2, 0.3, -0.1])
        np.testing.assert_allclose(B.compose(A).point(p), B.point(A.point(p)), atol=1e-13)

    def test_reject_reflection_scaling_and_nan(self):
        for R in [np.diag([-1, 1, 1]), np.diag([2, 1, 1]),
                  np.full((3, 3), float("nan"))]:
            with self.assertRaises(ValueError):
                RigidTransform(rotation=R)
        with self.assertRaises(ValueError):
            RigidTransform(translation=[0, float("inf"), 0])

    def test_frame_conversion_flips_products_as_expected(self):
        T = FORWARD_PORT_UP_TO_FORWARD_STARBOARD_DOWN
        np.testing.assert_allclose(T.rotation @ T.rotation, np.eye(3))
        I = np.array([[8, -1, -2], [-1, 9, -3], [-2, -3, 10]], dtype=float)
        result = T.inertia_cg(I)
        # x-y and x-z entries change signs, y-z entry does not.
        np.testing.assert_allclose(result, [[8, 1, 2], [1, 9, -3], [2, -3, 10]])
        np.testing.assert_allclose(T.inertia_cg(result), I)

    def test_eigenvalues_and_trace_rotation_invariant(self):
        T = RigidTransform.from_axis_angle([1, 2, 3], 0.53)
        I = np.array([[13, -1, 2], [-1, 16, -3], [2, -3, 20]])
        J = T.inertia_cg(I)
        np.testing.assert_allclose(np.linalg.eigvalsh(J), np.linalg.eigvalsh(I))
        self.assertAlmostEqual(np.trace(J), np.trace(I))


class MassModelTransformTests(unittest.TestCase):

    @staticmethod
    def model():
        p = MassProperties()
        p.add_point_mass(2, [1, 3, -1], label="person")
        p.add_box(18, [4, -2, 1], [2, 3, 5], label="box")
        p.add_component(MassComponent(
            4, [0, 1, 2], np.array([[5., -1, 0], [-1, 5, 0], [0, 0, 4]]), "assembly"))
        return p

    def test_collection_equals_result_transform(self):
        source = self.model()
        T = RigidTransform.from_axis_angle([0, 1, 1], 0.6, translation=[7, -4, 5])
        original = source.calculate()
        directly = original.transformed(T)
        transformed = source.transformed(T).calculate()
        self.assertEqual(directly.mass, transformed.mass)
        np.testing.assert_allclose(directly.center_of_gravity, transformed.center_of_gravity, atol=1e-13)
        np.testing.assert_allclose(directly.inertia_tensor, transformed.inertia_tensor, atol=1e-12)
        np.testing.assert_allclose(source.calculate().center_of_gravity,
                                   original.center_of_gravity)

    def test_inertia_at_transformed_origin(self):
        model = self.model()
        T = RigidTransform.from_axis_angle([1, 2, 0], 0.33, translation=[2, 3, -1])
        origin_source = [2, -2, 0.5]
        Ia = model.calculate().inertia_at(origin_source)
        Ib = model.transformed(T).calculate().inertia_at(T.point(origin_source))
        np.testing.assert_allclose(Ib, T.inertia_cg(Ia), atol=1e-12)

    def test_pure_translation_invariance(self):
        model = self.model()
        T = RigidTransform(translation=[20, 50, -3])
        source = model.calculate()
        target = model.transformed(T).calculate()
        np.testing.assert_allclose(target.inertia_tensor, source.inertia_tensor)
        np.testing.assert_allclose(target.center_of_gravity,
                                   source.center_of_gravity + [20, 50, -3])

    def test_reflection_is_not_permitted_for_mass_model(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.transformed(RigidTransform(np.diag([-1, 1, 1])))

    def test_transform_type_validation(self):
        with self.assertRaises(TypeError):
            self.model().transformed(np.eye(3))
        with self.assertRaises(TypeError):
            self.model().calculate().transformed(np.eye(3))

    def test_label_and_mass_preservation(self):
        T = RigidTransform.from_axis_angle([0, 0, 1], 0.1)
        a, b = self.model().components, self.model().transformed(T).components
        self.assertEqual([x.label for x in a], [x.label for x in b])
        self.assertEqual([x.mass for x in a], [x.mass for x in b])


if __name__ == "__main__":
    unittest.main()
