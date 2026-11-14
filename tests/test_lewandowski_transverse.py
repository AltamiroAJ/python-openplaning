"""28C.3: independent characteristic determinant, eigenmodes, inertia signs."""
import unittest
from math import sqrt
from types import SimpleNamespace

import numpy as np

from openplaning.stability import (
    TransverseCoefficientState, evaluate_hull_derivatives, LewandowskiState,
    RollForceModel,
)
from openplaning.stability.transverse import (
    RigidInertia, linearize_transverse_system, assemble_lewandowski_2004,
    from_resolved_planing_boat,
)


class AlgebraicTests(unittest.TestCase):
    def setUp(self):
        self.m = 10.
        self.M = np.diag([-10., -20., -30.])
        self.D = np.diag([-20., -6., -15.])
        self.h = np.array([0., -80., 0.])

    def test_uncoupled_reference_eigenvalues(self):
        out = linearize_transverse_system(self.M, self.D, self.h, mass=10., beam=2.)
        reference = [-2., -0.5, -0.15+1j*sqrt(4-.15**2), -0.15-1j*sqrt(4-.15**2)]
        for expected in reference:
            self.assertTrue(any(abs(x-expected) < 1e-11 for x in out.eigenvalues))
        self.assertEqual(out.classification, 'stable_linear_model')
        np.testing.assert_allclose(np.poly(out.state_matrix),
                                   out.characteristic_polynomial/out.characteristic_polynomial[0],
                                   atol=1e-11)
        self.assertAlmostEqual(out.spectral_abscissa, -.15)

    def test_one_positive_root(self):
        D = self.D.copy(); D[0, 0] = 20.
        out = linearize_transverse_system(self.M, D, self.h, mass=10., beam=2.)
        self.assertEqual(out.classification, 'unstable_linear_model')
        self.assertAlmostEqual(out.spectral_abscissa, 2.)

    def test_marginal_case(self):
        D = self.D.copy(); D[2, 2] = 0
        out = linearize_transverse_system(self.M, D, self.h, mass=10., beam=2.)
        self.assertEqual(out.classification, 'marginal_or_unresolved')

    def test_root_tolerance(self):
        out = linearize_transverse_system(self.M, self.D, self.h,
                                          mass=10., beam=2., tolerance_per_second=.2)
        self.assertEqual(out.classification, 'marginal_or_unresolved')

    def test_singular_mass_rejected(self):
        M = self.M.copy(); M[1, :] = 0
        with self.assertRaises(ValueError):
            linearize_transverse_system(M, self.D, self.h, mass=10., beam=2.)

    def test_nonfinite_entries_rejected(self):
        M = self.M.copy(); M[1, 0] = np.inf
        with self.assertRaises(ValueError):
            linearize_transverse_system(M, self.D, self.h, mass=10., beam=2.)

    def test_denied_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            linearize_transverse_system(self.M, self.D, self.h,
                                        mass=10., beam=2., tolerance_per_second=-1.)

    def test_no_mutation_of_inputs(self):
        a,b,c=self.M.copy(),self.D.copy(),self.h.copy()
        out=linearize_transverse_system(a,b,c,mass=10.,beam=2.)
        np.testing.assert_array_equal(a,self.M)
        np.testing.assert_array_equal(b,self.D)
        np.testing.assert_array_equal(c,self.h)
        with self.assertRaises(ValueError): out.state_matrix[0,0]=1.

    def test_polynomial_independent_for_coupled_example(self):
        M = self.M.copy(); M[1,2]=4; M[2,1]=4; M[0,1]=2
        D = self.D.copy(); D[0,2]=-5; D[2,0]=3
        h = self.h.copy(); h[0]=2; h[2]=1
        out=linearize_transverse_system(M,D,h,mass=10.,beam=2.)
        np.testing.assert_allclose(np.roots(out.characteristic_polynomial).real.sum(),
                                   out.eigenvalues.real.sum(),atol=1e-10)

    def test_roll_rates_are_per_second(self):
        out=linearize_transverse_system(self.M,self.D,self.h,mass=10.,beam=2.)
        self.assertAlmostEqual(max(out.oscillation_frequencies_hz),
                               sqrt(4-.15**2)/(2*np.pi))


class SourceAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.hstate=TransverseCoefficientState(speed=15.,rho=1025.,g=9.81,beam=2.,
             L_K=5.,L_C=4.,draft_transom=.3,beta_deg=15.,tau_deg=5.,vcg=.7,lcg=2.)
        self.rstate=LewandowskiState(speed=15.,rho=1025.,g=9.81,beam=2.,
             L_K=5.,L_C=4.,L_C2=0.,beta_deg=15.,tau_deg=5.,vcg=.7,
             lcg=2.,lcp=2.4)
        self.hull=evaluate_hull_derivatives(self.hstate,source='lew2004_table63')
        self.roll=RollForceModel(self.rstate,source_variant='lew1996_eq27').derivatives()
        self.rigid=RigidInertia(1500.,4000.,12000.,-100.)

    def test_assembly_source_form_and_inertia_sign(self):
        out=assemble_lewandowski_2004(self.hull,self.roll,self.rigid,roll_state=self.rstate)
        self.assertAlmostEqual(out.acceleration_matrix[0,0],self.hull['Y_v_dot']-1500)
        self.assertAlmostEqual(out.acceleration_matrix[1,2],self.hull['K_r_dot']+100)
        self.assertAlmostEqual(out.acceleration_matrix[2,1],self.hull['N_phi_ddot']+100)
        self.assertAlmostEqual(out.rate_matrix[0,2],self.hull['Y_r']-1500*15)
        self.assertAlmostEqual(out.heel_column[1],self.roll.K_phi_total)
        self.assertTrue(np.all(np.isfinite(out.eigenvalues)))
        self.assertEqual(len(out.eigenvalues),4)

    def test_no_changes_to_derivative_values(self):
        before=dict(self.hull.values)
        assemble_lewandowski_2004(self.hull,self.roll,self.rigid,roll_state=self.rstate)
        self.assertEqual(before,self.hull.values)

    def test_reject_1997_dimensional_variant(self):
        alternate=evaluate_hull_derivatives(self.hstate,source='lew1997_appendix')
        with self.assertRaises(ValueError):
            assemble_lewandowski_2004(alternate,self.roll,self.rigid,roll_state=self.rstate)

    def test_reject_dynroll(self):
        alternate=evaluate_hull_derivatives(self.hstate,source='dynroll_prototype')
        with self.assertRaises(ValueError):
            assemble_lewandowski_2004(alternate,self.roll,self.rigid,roll_state=self.rstate)

    def test_reject_roll_variant(self):
        altered=RollForceModel(self.rstate,source_variant='dynroll_1997_prototype').derivatives()
        with self.assertRaises(ValueError):
            assemble_lewandowski_2004(self.hull,altered,self.rigid,roll_state=self.rstate)

    def test_reject_mixed_equilibrium_states(self):
        different=LewandowskiState(**{**self.rstate.__dict__,'speed':14.})
        with self.assertRaisesRegex(ValueError,'inconsistent source states'):
            assemble_lewandowski_2004(self.hull,self.roll,self.rigid,roll_state=different)

    def test_reject_missing_inertia(self):
        with self.assertRaises(ValueError):
            assemble_lewandowski_2004(self.hull,self.roll,RigidInertia(1500,0,100,0),roll_state=self.rstate)

    def test_rigid_tensor_raw_product_sign(self):
        inertia=np.array([[4000,0,-100],[0,5000,0],[-100,0,12000]],float)
        obj=SimpleNamespace(mass=1500,center_of_gravity=np.array([2,0,.7]),inertia_tensor=inertia)
        rigid=RigidInertia.from_snapshot(obj)
        self.assertEqual(rigid.Ixz_tensor,-100)
        self.assertEqual(rigid.Ixz_book_raw,100)

    def test_reject_asymmetric_tensor(self):
        inertia=np.array([[4000,50,-100],[50,5000,0],[-100,0,12000]],float)
        obj=SimpleNamespace(mass=1500,center_of_gravity=np.array([2,0,.7]),inertia_tensor=inertia)
        with self.assertRaisesRegex(ValueError,'asymmetric'):
            RigidInertia.from_snapshot(obj)

    def test_reject_lateral_cg(self):
        inertia=np.diag([4000,5000,12000])
        obj=SimpleNamespace(mass=1500,center_of_gravity=np.array([2,.1,.7]),inertia_tensor=inertia)
        with self.assertRaisesRegex(ValueError,'lateral CG'):
            RigidInertia.from_snapshot(obj)

    def test_facade_no_mutation_and_no_steady_trim(self):
        # Minimal resolved PlaningBoat-shaped fake: deliberately has no solver methods.
        beam=2.; m=1500.; Iyy=5000.
        boat=SimpleNamespace(
            speed=15.,rho=1025.,g=9.81,beam=beam,L_K=5.,L_C=4.,L_C2=0.,T=.3,
            beta=15.,tau=5.,vcg=.7,lcg=2.,lcp=2.4,
            weight=m*9.81,r_g=sqrt(Iyy/m))
        mass=SimpleNamespace(mass=m,center_of_gravity=np.array([2.,0.,.7]),
                  inertia_tensor=np.diag([4000.,Iyy,12000.]))
        boat.get_mass_properties_snapshot=lambda: mass
        original={k:v for k,v in vars(boat).items() if k!='get_mass_properties_snapshot'}
        out=from_resolved_planing_boat(boat)
        self.assertEqual(len(out.eigenvalues),4)
        for k,v in original.items(): self.assertEqual(getattr(boat,k),v)


if __name__=='__main__':
    unittest.main()
