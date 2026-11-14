"""Independent source identities for 28C.7 (no changes to earlier solvers)."""
import unittest
from dataclasses import replace
from math import cos, radians, sin, tan
import numpy as np

from openplaning.stability.lewandowski import LewandowskiState,RollForceModel
from openplaning.stability.source_audit import audit_roll_sources, _source_static_2004


class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.s = LewandowskiState(speed=12.,rho=1025.,g=9.81,beam=2.2,
             beta_deg=18.,tau_deg=3.0916,L_K=6.051,L_C=5.0,L_C2=.2,
             vcg=.65,lcg=2.25,lcp=2.6)

    def test_1996_eq27_force_zero_positive(self):
        self.assertGreater(audit_roll_sources(self.s).side_force_zero_N,0.)

    def test_exact_projection_chain_rule(self):
        a=audit_roll_sources(self.s)
        np.testing.assert_allclose(a.y_phi_1996_projection_minus_1996_rate_N_per_rad,
             a.y_phi_expected_geometric_term_N_per_rad,rtol=1e-7,atol=1e-5)

    def test_two_force_definitions_not_silently_equal(self):
        a=audit_roll_sources(self.s)
        self.assertNotAlmostEqual(a.y_phi_full_projection_N_per_rad,
                                  a.y_phi_2004_table63_N_per_rad,places=3)

    def test_2004_table62_k_phi_direct_independent(self):
        a=audit_roll_sources(self.s)
        h2=RollForceModel(self.s).lever_arm()
        self.assertAlmostEqual(a.k_phi_2004_table62_Nm_per_rad,
                               2*h2*a.side_force_rate_2004_table62_N_per_rad,places=6)

    def test_1996_k_phi_independent_derivative(self):
        a=audit_roll_sources(self.s)
        h2=RollForceModel(self.s).lever_arm()
        np.testing.assert_allclose(a.k_phi_dynamic_1996_Nm_per_rad,
            2*h2*a.side_force_rate_five_point_N_per_rad,rtol=1e-8)

    def test_yaw_lever_arm_identity_both_variants(self):
        a=audit_roll_sources(self.s)
        self.assertAlmostEqual(a.n_phi_full_projection_Nm_per_rad,
             a.y_phi_full_projection_N_per_rad*(self.s.lcp-self.s.lcg),places=8)
        self.assertAlmostEqual(a.n_phi_2004_table63_Nm_per_rad,
             a.y_phi_2004_table63_N_per_rad*(self.s.lcp-self.s.lcg),places=8)

    def test_printed_2004_yphi_exact(self):
        a=audit_roll_sources(self.s)
        self.assertAlmostEqual(a.y_phi_2004_table63_N_per_rad,
              2*a.side_force_rate_2004_table62_N_per_rad*sin(radians(self.s.beta_deg)),places=5)

    def test_2004_phi_not_1996_phi_exactly(self):
        a=audit_roll_sources(self.s)
        self.assertNotAlmostEqual(a.side_force_rate_2004_table62_N_per_rad,
                                  a.side_force_rate_five_point_N_per_rad,places=2)

    def test_analytic_2004_delta_lift(self):
        a=audit_roll_sources(self.s)
        expected=.25*self.s.rho*self.s.g*self.s.lambda_w**2*2.2**3*sin(2*radians(self.s.tau_deg))
        self.assertAlmostEqual(a.delta_static_lift_2004_N,expected,places=5)

    def test_fsw_high_speed_branch(self):
        a=audit_roll_sources(replace(self.s,speed=60.))
        self.assertEqual(a.fsw_2004,1.)
        self.assertEqual(a.side_wetting_condition,'high_speed_fsw_equal_one')

    def test_fsw_low_speed_branch(self):
        s=replace(self.s,speed=1.)
        a=audit_roll_sources(s)
        self.assertEqual(a.side_wetting_condition,'low_speed_fsw_includes_L_C2')
        self.assertAlmostEqual(a.fsw_2004,1+.603*s.L_C2/(s.lambda_w*s.beam))

    def test_fsw_without_lc2_no_correction(self):
        s=replace(self.s,speed=1.,L_C2=0.)
        self.assertEqual(audit_roll_sources(s).fsw_2004,1.)

    def test_static_formula_independent(self):
        a=audit_roll_sources(self.s)
        b,Lk,Lc=self.s.beam,self.s.L_K,self.s.L_C
        KB=b*tan(radians(self.s.beta_deg))*(.5+Lc/Lk)/6
        expected=.624*a.fsw_2004*(-self.s.rho*self.s.g*b**3*(Lk+3*Lc)/48+
                                   (self.s.vcg-KB)*a.delta_static_lift_2004_N)
        self.assertAlmostEqual(a.k_phi_static_2004_Nm_per_rad,expected,places=6)

    def test_legacy_static_is_not_source_static(self):
        a=audit_roll_sources(self.s)
        self.assertNotAlmostEqual(a.k_phi_static_2004_Nm_per_rad,
                                   a.k_phi_static_legacy_Nm_per_rad,places=3)

    def test_zero_step_rejected(self):
        with self.assertRaises(ValueError):audit_roll_sources(self.s,step_rad=0.)

    def test_large_step_rejected(self):
        with self.assertRaises(ValueError):audit_roll_sources(self.s,step_rad=.01)

    def test_source_flags_nonempty(self):
        self.assertTrue(audit_roll_sources(self.s).source_consistency_flags)

    def test_1996_lambda_geometry_residual_computed(self):
        a=audit_roll_sources(self.s)
        expected=self.s.lambda_w-(self.s.L_K/self.s.beam-tan(radians(self.s.beta_deg)) /
                                  (2*np.pi*tan(radians(self.s.tau_deg))))
        self.assertAlmostEqual(a.lambda_geometry_residual,expected)

    def test_no_state_mutation(self):
        before=vars(self.s).copy();audit_roll_sources(self.s)
        self.assertEqual(before,vars(self.s))

    def test_five_point_and_central_converge(self):
        a=audit_roll_sources(self.s,step_rad=1e-5)
        np.testing.assert_allclose(a.side_force_rate_N_per_rad,
                                    a.side_force_rate_five_point_N_per_rad,rtol=1e-7)

    def test_static_source_independent_of_lcp(self):
        a=audit_roll_sources(self.s)
        b=audit_roll_sources(replace(self.s,lcp=self.s.lcp+.5))
        self.assertEqual(a.k_phi_static_2004_Nm_per_rad,b.k_phi_static_2004_Nm_per_rad)

    def test_nonlinear_sources_do_not_enter_solver_automatically(self):
        a=audit_roll_sources(self.s)
        self.assertFalse(hasattr(a,'state_matrix'))

if __name__=='__main__':unittest.main()
