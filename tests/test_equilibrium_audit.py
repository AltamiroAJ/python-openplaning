"""28C.8: falsifiable equilibrium-vs-Jacobian and source/geometry splitting."""
import unittest
from dataclasses import replace
from math import cos, radians

import numpy as np

from openplaning.stability.lewandowski import LewandowskiState, RollForceModel
from openplaning.stability.source_audit import audit_roll_sources
from openplaning.stability.equilibrium_audit import (
    _table_2004_force_rate, decompose_lateral_discrepancy,
    test_equilibrium_derivative,
)


class EquilibriumAuditTests(unittest.TestCase):
    def setUp(self):
        self.s = LewandowskiState(
            speed=12., rho=1025., g=9.81, beam=2.2,
            beta_deg=18., tau_deg=3.0916,
            L_K=6.051, L_C=4.65, L_C2=0.,
            vcg=.65, lcg=2.25, lcp=2.27,
        )

    def test_balance_at_phi_zero(self):
        r = test_equilibrium_derivative(self.s)
        self.assertAlmostEqual(r.total_force_zero_N, 0., places=12)

    def test_balance_does_not_eliminate_jacobian(self):
        r = test_equilibrium_derivative(self.s)
        self.assertAlmostEqual(r.total_rate_N_per_rad, r.y_dynamic_rate_N_per_rad)
        self.assertGreater(abs(r.total_rate_N_per_rad), 1.)

    def test_constant_counterforce_only_affects_reference(self):
        r = test_equilibrium_derivative(self.s, static_heel_rate_N_per_rad=0.)
        self.assertEqual(r.static_force_zero_for_balance_N, -r.y_dynamic_zero_N)
        self.assertAlmostEqual(r.total_rate_N_per_rad, r.y_dynamic_rate_N_per_rad)

    def test_prescribed_slope_cancels_only_geometric_term(self):
        original = test_equilibrium_derivative(self.s)
        changed = test_equilibrium_derivative(self.s,
            static_heel_rate_N_per_rad=original.cancellation_requires_rate_N_per_rad)
        decomp = decompose_lateral_discrepancy(self.s)
        self.assertAlmostEqual(changed.total_rate_N_per_rad,
                               decomp.y1996_normal_rate, delta=1e-4)
        self.assertAlmostEqual(changed.total_force_zero_N, 0., places=12)

    def test_a_force_slope_must_be_external_not_inferred(self):
        r = test_equilibrium_derivative(self.s, static_heel_rate_N_per_rad=-2000.)
        self.assertAlmostEqual(r.total_rate_N_per_rad,
                               r.y_dynamic_rate_N_per_rad-2000.)

    def test_geometric_term_equals_2F0cos_beta(self):
        result = decompose_lateral_discrepancy(self.s)
        audit = audit_roll_sources(self.s)
        expected = 2*audit.side_force_zero_N*cos(radians(self.s.beta_deg))
        self.assertAlmostEqual(result.projection_term, expected, delta=.002)

    def test_residual_decomposes_exactly(self):
        d = decompose_lateral_discrepancy(self.s)
        self.assertAlmostEqual(d.total_difference, d.recomposed_difference, delta=1e-7)
        self.assertAlmostEqual(d.total_difference,
           d.projection_term+d.source_difference_at_matched_lambda+d.wet_length_difference,delta=1e-7)

    def test_2004_baseline_reproduces_28c7(self):
        d = decompose_lateral_discrepancy(self.s)
        prior = audit_roll_sources(self.s)
        self.assertAlmostEqual(d.y2004_at_sav,
                               prior.y_phi_2004_table63_N_per_rad,delta=1e-8)

    def test_source_is_matched_at_same_lambda_before_comparison(self):
        d = decompose_lateral_discrepancy(self.s)
        self.assertAlmostEqual(d.source_difference_at_matched_lambda,
            d.y1996_normal_rate-d.y2004_at_eq26)

    def test_1996_side_force_independent_of_supplied_LC(self):
        first=RollForceModel(self.s)
        alt=RollForceModel(replace(self.s,L_C=self.s.L_C-.6))
        for angle in (-.01,0.,.01):
            self.assertAlmostEqual(first.side_normal_force(self.s.beta_deg + np.rad2deg(angle)),
                                   alt.side_normal_force(self.s.beta_deg + np.rad2deg(angle)),delta=1e-8)

    def test_2004_table_changes_with_supplied_LC(self):
        d=decompose_lateral_discrepancy(self.s)
        alt=decompose_lateral_discrepancy(replace(self.s,L_C=self.s.L_C-.6))
        self.assertNotAlmostEqual(d.y2004_at_sav,alt.y2004_at_sav,delta=1e-5)
        self.assertAlmostEqual(d.y1996_normal_rate,alt.y1996_normal_rate,delta=1e-5)

    def test_when_lengths_match_geometry_component_zero(self):
        d=decompose_lateral_discrepancy(self.s)
        state=replace(self.s,L_C=d.lc_eq26_m)
        matched=decompose_lateral_discrepancy(state)
        self.assertAlmostEqual(matched.wet_length_difference,0.,delta=1e-9)
        self.assertAlmostEqual(matched.lambda_sav,matched.lambda_eq26,delta=1e-12)

    def test_counterfactual_table_function_well_defined(self):
        with self.assertRaises(ValueError): _table_2004_force_rate(self.s,0.)
        with self.assertRaises(ValueError): _table_2004_force_rate(self.s,float('nan'))

    def test_nan_static_rate_rejected(self):
        with self.assertRaises(ValueError):test_equilibrium_derivative(self.s,static_heel_rate_N_per_rad=float('nan'))

    def test_invalid_input_type_rejected(self):
        with self.assertRaises(TypeError):decompose_lateral_discrepancy(None)

    def test_n_moment_residual_uses_book_lever(self):
        d=decompose_lateral_discrepancy(self.s)
        self.assertAlmostEqual(d.n_difference,d.total_difference*(self.s.lcp-self.s.lcg),delta=1e-7)

    def test_no_state_mutation(self):
        d0=dict(self.s.__dict__)
        decompose_lateral_discrepancy(self.s)
        test_equilibrium_derivative(self.s)
        self.assertEqual(d0,self.s.__dict__)

    def test_genuine_static_slope_cannot_be_obtained_from_static_lift(self):
        # Two total-force laws with IDENTICAL Y(0)=0, but different Jacobians.
        a=test_equilibrium_derivative(self.s,static_heel_rate_N_per_rad=0.)
        b=test_equilibrium_derivative(self.s,static_heel_rate_N_per_rad=-a.geometric_rate_N_per_rad)
        self.assertAlmostEqual(a.total_force_zero_N,b.total_force_zero_N)
        self.assertNotAlmostEqual(a.total_rate_N_per_rad,b.total_rate_N_per_rad,delta=1.)


if __name__=='__main__': unittest.main()
