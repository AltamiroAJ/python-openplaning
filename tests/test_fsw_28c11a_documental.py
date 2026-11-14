"""28C.11a: literal-source regression, WITHOUT changing the hydrodynamic model.

2004 Table 6.2 (visually checked): f_sw = 1 + 0.603 L_C2/(lambda B)
when U**2 is below the tabulated threshold. The exact equality is not
specified by the published strict inequalities; code chooses the low branch.
"""
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from openplaning.stability.lewandowski import LewandowskiState
from openplaning.stability import source_audit as source


class FswDocumentaryTests(unittest.TestCase):
    def setUp(self):
        self.state = LewandowskiState(
            speed=1., rho=1025., g=9.81, beam=2.2,
            beta_deg=18., tau_deg=3.0916,
            L_K=6.051, L_C=5., L_C2=.2,
            vcg=.65, lcg=2.25, lcp=2.6,
        )

    def test_literal_low_branch_factor(self):
        # Independently calculated, fixed reference, NOT a call to _source_fsw.
        factor, regime = source._source_fsw(self.state)
        self.assertEqual(regime, 'low_speed_fsw_includes_L_C2')
        self.assertAlmostEqual(factor, 1.0218260790878653, places=13)
        self.assertAlmostEqual(factor-1., .603*.2/(2.5115909090909088*2.2), places=14)

    def test_literal_high_branch_factor(self):
        factor, regime = source._source_fsw(replace(self.state, speed=60.))
        self.assertEqual(factor, 1.)
        self.assertEqual(regime, 'high_speed_fsw_equal_one')

    def test_literal_static_moment_low_branch(self):
        static, delta, factor, regime = source._source_static_2004(self.state)
        # Independent literal reference computed from printed 2004 Table 6.2/6.3.
        self.assertAlmostEqual(static, -24235.07487158906, places=7)
        self.assertAlmostEqual(factor, 1.0218260790878653, places=13)
        self.assertEqual(regime, 'low_speed_fsw_includes_L_C2')

    def test_literal_static_moment_high_branch(self):
        high = replace(self.state, speed=60.)
        static, delta, factor, regime = source._source_static_2004(high)
        self.assertEqual(factor, 1.)
        # Static force expression depends on equilibrium geometry, not U directly.
        self.assertEqual(static, -23717.416659812145)

    def test_factor_uses_L_C2_not_L_C(self):
        changed_lc2 = replace(self.state, L_C2=.4)
        f1, _ = source._source_fsw(self.state)
        f2, _ = source._source_fsw(changed_lc2)
        self.assertAlmostEqual(f2-1, 2*(f1-1), places=13)
        self.assertEqual(source._source_fsw(replace(self.state,L_C2=0.))[0], 1.)

    def test_boundary_equality_is_code_convention_not_source_claim(self):
        # Algebraically exact threshold for the SAME code expression using
        # a mock of trigonometry (not a physically admissible boat). This
        # exercises the equality path without roundoff from sqrt(threshold).
        s = replace(self.state, speed=2., g=3., beam=2., L_K=4., L_C=4.,
                    L_C2=.5)
        self.assertEqual(s.lambda_w, 2.)
        with patch.object(source, 'tan', side_effect=lambda angle: 0. if angle == math.radians(s.beta_deg) else 1.), \
             patch.object(source, 'sin', return_value=1.):
            threshold = s.g*s.beam*(s.lambda_w-0.)/(3*1.)
            self.assertEqual(threshold, 4.)
            self.assertEqual(s.speed*s.speed, threshold)
            at, label = source._source_fsw(s)
            below, _ = source._source_fsw(replace(s, speed=math.nextafter(2.,0.)))
            above, label_above = source._source_fsw(replace(s, speed=math.nextafter(2., math.inf)))
        self.assertEqual(at, 1.075375)
        self.assertEqual(below, at)
        self.assertEqual(above, 1.)
        self.assertEqual(label, 'low_speed_fsw_includes_L_C2')
        self.assertEqual(label_above, 'high_speed_fsw_equal_one')

    def test_no_roll_source_1996_or_2004_solver_side_effect(self):
        # This is a read-only test: no solver is invoked or altered.
        before = dict(vars(self.state))
        _ = source._source_static_2004(self.state)
        self.assertEqual(vars(self.state), before)


if __name__ == '__main__':
    unittest.main()
