"""28C.5: frame-resolved *hypothesis*, not an experimental stability validation."""
import unittest
from types import SimpleNamespace
from math import sqrt
import numpy as np

from openplaning.stability import (
    TransverseCoefficientState, evaluate_hull_derivatives,
    LewandowskiState, RollForceModel,
)
from openplaning.stability.transverse import RigidInertia, assemble_lewandowski_2004
from openplaning.stability.frame_audit import (
    DescriptorTransform, book_xz_raw_from_fpu_tensor,
)
from openplaning.stability.frame_resolved import (
    assemble_lewandowski_2004_fsd, from_resolved_planing_boat_fsd,
    compare_legacy_and_fsd,
)
from openplaning.stability.transverse import linearize_transverse_system


class FrameResolvedTests(unittest.TestCase):
    def setUp(self):
        self.hstate = TransverseCoefficientState(
            speed=12., rho=1025., g=9.81, beam=2.2,
            L_K=6.05, L_C=5.0, draft_transom=.28,
            beta_deg=18., tau_deg=3.0916, vcg=.65, lcg=2.25)
        self.rstate = LewandowskiState(
            speed=12., rho=1025., g=9.81, beam=2.2,
            L_K=6.05, L_C=5.0, L_C2=0., beta_deg=18.,
            tau_deg=3.0916, vcg=.65, lcg=2.25, lcp=2.6)
        self.hull = evaluate_hull_derivatives(self.hstate)
        self.roll = RollForceModel(self.rstate).derivatives()
        self.rigid = RigidInertia(1600., 4000., 6800., 144.)

    def run_fsd(self, rigid=None, hull=None, roll=None, roll_state=None):
        return assemble_lewandowski_2004_fsd(
            self.hull if hull is None else hull,
            self.roll if roll is None else roll,
            self.rigid if rigid is None else rigid,
            roll_state=self.rstate if roll_state is None else roll_state,
        )

    def test_book_pxz_mapping_independent(self):
        I = np.array([[4000., 0, 144.], [0, 4500., 0], [144., 0, 6800.]])
        self.assertEqual(book_xz_raw_from_fpu_tensor(I), 144.)
        self.assertEqual(self.rigid.Ixz_book_raw, -144.)
        self.assertEqual(self.run_fsd().acceleration_matrix[1,2], 144.)
        self.assertEqual(self.run_fsd().acceleration_matrix[2,1], 144.)

    def test_mass_and_hull_terms_preserved(self):
        out = self.run_fsd()
        self.assertAlmostEqual(out.acceleration_matrix[0,0], self.hull['Y_v_dot']-1600.)
        self.assertAlmostEqual(out.acceleration_matrix[1,1], self.hull['K_phi_ddot']-4000.)
        self.assertAlmostEqual(out.acceleration_matrix[2,2], self.hull['N_r_dot']-6800.)
        self.assertAlmostEqual(out.rate_matrix[0,2], self.hull['Y_r']-1600.*12.)
        self.assertAlmostEqual(out.rate_matrix[1,0], self.hull['K_v'])

    def test_roll_heel_column_transformed_once(self):
        out = self.run_fsd()
        np.testing.assert_allclose(out.heel_column,
            [-self.roll.Y_phi_dynamic, self.roll.K_phi_total,
             -self.roll.N_phi_dynamic])

    def test_book_vs_legacy_distinct_and_diagnosed(self):
        old = assemble_lewandowski_2004(self.hull, self.roll, self.rigid,
                                         roll_state=self.rstate)
        new = self.run_fsd()
        self.assertEqual(old.acceleration_matrix[1,2], -144.)
        self.assertEqual(new.acceleration_matrix[1,2], +144.)
        np.testing.assert_allclose(old.rate_matrix, new.rate_matrix)
        self.assertGreater(np.max(np.abs(new.state_matrix-old.state_matrix)), 0)
        self.assertEqual(len(new.eigenvalues), 4)

    def test_fsd_fpu_full_descriptor_similarity(self):
        fsd = self.run_fsd()
        T = DescriptorTransform.physical_fpu_to_fsd()  # self-inverse
        m,d,h = T.descriptor(fsd.acceleration_matrix, fsd.rate_matrix, fsd.heel_column)
        fpu = linearize_transverse_system(m,d,h,mass=1600.,beam=2.2)
        np.testing.assert_allclose(fpu.state_matrix,T.state_similarity(fsd.state_matrix),atol=1e-12)
        np.testing.assert_allclose(np.sort_complex(fpu.eigenvalues),
                                   np.sort_complex(fsd.eigenvalues),atol=1e-11)
        np.testing.assert_allclose(
            fpu.characteristic_polynomial/fpu.characteristic_polynomial[0],
            fsd.characteristic_polynomial/fsd.characteristic_polynomial[0],
            atol=1e-10)

    def test_legacy_unchanged_when_fsd_called(self):
        before=assemble_lewandowski_2004(self.hull,self.roll,self.rigid,roll_state=self.rstate)
        self.run_fsd()
        after=assemble_lewandowski_2004(self.hull,self.roll,self.rigid,roll_state=self.rstate)
        np.testing.assert_allclose(before.state_matrix,after.state_matrix,atol=0,rtol=0)

    def test_zero_pxz_is_neutral_in_kinetic_cross_term(self):
        out = self.run_fsd(RigidInertia(1600.,4000.,6800.,0.))
        self.assertEqual(out.acceleration_matrix[1,2],self.hull['K_r_dot'])
        self.assertEqual(out.acceleration_matrix[2,1],self.hull['N_phi_ddot'])

    def test_asymmetric_same_mass_opposite_pxz_physical_models(self):
        plus = self.run_fsd(RigidInertia(1600.,4000.,6800.,144.))
        minus = self.run_fsd(RigidInertia(1600.,4000.,6800.,-144.))
        self.assertEqual(plus.acceleration_matrix[1,2],144.)
        self.assertEqual(minus.acceleration_matrix[1,2],-144.)
        self.assertGreater(np.linalg.norm(plus.state_matrix-minus.state_matrix),1e-9)

    def test_published_1997_variant_denied(self):
        old = evaluate_hull_derivatives(self.hstate,source='lew1997_appendix')
        with self.assertRaises(ValueError):self.run_fsd(hull=old)

    def test_dynroll_variant_denied(self):
        old = evaluate_hull_derivatives(self.hstate,source='dynroll_prototype')
        with self.assertRaises(ValueError):self.run_fsd(hull=old)

    def test_roll_force_prototype_denied(self):
        r = RollForceModel(self.rstate,source_variant='dynroll_1997_prototype').derivatives()
        with self.assertRaises(ValueError):self.run_fsd(roll=r)

    def test_inconsistent_states_denied(self):
        other=LewandowskiState(**{**self.rstate.__dict__,'speed':14.})
        with self.assertRaises(ValueError):self.run_fsd(roll_state=other)

    def test_invalid_inputs_denied(self):
        with self.assertRaises(TypeError):assemble_lewandowski_2004_fsd(None,self.roll,self.rigid,roll_state=self.rstate)

    def _boat(self):
        obj=SimpleNamespace(
            speed=self.hstate.speed, rho=self.hstate.rho,g=self.hstate.g,
            beam=self.hstate.beam, L_K=self.hstate.L_K,L_C=self.hstate.L_C,
            L_C2=self.rstate.L_C2, T=self.hstate.draft_transom,
            beta=self.hstate.beta_deg,tau=self.hstate.tau_deg,
            vcg=self.hstate.vcg,lcg=self.hstate.lcg,lcp=self.rstate.lcp,
            weight=1600*9.81,r_g=sqrt(5000/1600),
        )
        snap=SimpleNamespace(
            mass=1600.,center_of_gravity=np.array([2.25,0,.65]),
            inertia_tensor=np.array([[4000.,0,144.],[0,5000.,0],[144.,0,6800.]]))
        obj.get_mass_properties_snapshot=lambda:snap
        return obj

    def test_facade_does_not_mutate_boat(self):
        boat=self._boat()
        before={k:v for k,v in vars(boat).items() if k!='get_mass_properties_snapshot'}
        out=from_resolved_planing_boat_fsd(boat)
        self.assertTrue(np.all(np.isfinite(out.eigenvalues)))
        for k,v in before.items():self.assertEqual(v,getattr(boat,k))

    def test_comparison_api_returns_delta_not_target(self):
        compare=compare_legacy_and_fsd(self._boat())
        self.assertAlmostEqual(compare.spectral_abscissa_difference,
                               compare.fsd.spectral_abscissa-compare.legacy.spectral_abscissa)
        self.assertGreaterEqual(compare.eigenvalue_maximum_assignment_difference,0)
        self.assertEqual(len(compare.legacy.eigenvalues),len(compare.fsd.eigenvalues))

    def test_assumptions_flag_hypotheses(self):
        words=' '.join(self.run_fsd().assumptions)
        self.assertIn('FSD',words)
        self.assertIn('Unvalidated',words)
        self.assertIn('appendage',words)


if __name__ == '__main__': unittest.main()
