import unittest
from types import SimpleNamespace
import numpy as np
from openplaning.mass_properties import MassProperties, MassComponent


class MassPropertiesTests(unittest.TestCase):
    def test_empty_model(self):
        with self.assertRaises(ValueError): MassProperties().calculate()

    def test_single_point(self):
        p = MassProperties(); p.add_point_mass(25, (3, -1, 2))
        r = p.calculate()
        self.assertEqual(r.mass, 25)
        np.testing.assert_allclose(r.center_of_gravity, [3, -1, 2])
        np.testing.assert_allclose(r.inertia_tensor, np.zeros((3,3)))

    def test_two_point_masses(self):
        p=MassProperties(); p.add_point_mass(2, (-1, 0, 0)); p.add_point_mass(2, (1, 0, 0))
        r=p.calculate(); np.testing.assert_allclose(r.inertia_tensor, np.diag([0,4,4]))

    def test_asymmetric_product_of_inertia(self):
        p=MassProperties(); p.add_point_mass(1, (1,1,0)); p.add_point_mass(1, (-1,-1,0))
        r=p.calculate(); np.testing.assert_allclose(r.inertia_tensor, [[2,-2,0],[-2,2,0],[0,0,4]])

    def test_uniform_box(self):
        p=MassProperties(); p.add_box(12, (0,0,0), (2,3,4))
        np.testing.assert_allclose(p.calculate().inertia_tensor, np.diag([25,20,13]))

    def test_rotated_box(self):
        p=MassProperties(); R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        p.add_box(12, (0,0,0), (2,3,4), rotation=R)
        np.testing.assert_allclose(p.calculate().inertia_tensor, np.diag([20,25,13]))

    def test_parallel_axis(self):
        p=MassProperties(); p.add_point_mass(3, (2,0,0))
        np.testing.assert_allclose(p.calculate().inertia_at((0,0,0)),np.diag([0,12,12]))

    def test_translation_invariance(self):
        def sample(shift):
            p=MassProperties(); p.add_point_mass(3, np.array([1,2,3])+shift)
            p.add_box(5,np.array([6,-2,0])+shift,(2,3,4)); return p.calculate()
        a=sample(np.zeros(3)); b=sample(np.array([50,-30,21]))
        np.testing.assert_allclose(a.inertia_tensor,b.inertia_tensor,atol=1e-11)
        np.testing.assert_allclose(b.center_of_gravity-a.center_of_gravity,[50,-30,21])

    def test_reject_invalid_mass_and_dimensions(self):
        p=MassProperties()
        for m in (0,-1,float('nan')):
            with self.assertRaises(ValueError): p.add_point_mass(m,(0,0,0))
        with self.assertRaises(ValueError): p.add_box(1,(0,0,0),(2,-3,4))

    def test_reject_unphysical_tensor(self):
        with self.assertRaises(ValueError): MassComponent(1,(0,0,0),np.diag([1,1,5]))
        with self.assertRaises(ValueError): MassComponent(1,(0,0,0),np.diag([1,-1,1]))

    def test_reject_improper_rotation(self):
        p=MassProperties()
        with self.assertRaises(ValueError): p.add_box(1,(0,0,0),(1,2,3),rotation=np.diag([-1,1,1]))

    def test_planingboat_consistency_nonmutating(self):
        p=MassProperties(); p.add_box(12,(5,0,1),(2,3,4))
        r=p.calculate(); expected_rg=np.sqrt(r.inertia_tensor[1,1]/r.mass)
        b=SimpleNamespace(weight=12*9.81,lcg=5,vcg=1,r_g=expected_rg,g=9.81)
        original=b.__dict__.copy(); report=p.compare_to_planing_boat(b)
        self.assertTrue(all(v[2] for v in report.values()))
        self.assertEqual(b.__dict__, original)

if __name__ == '__main__': unittest.main()
