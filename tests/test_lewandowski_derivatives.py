"""Unit and source regression tests for Lewandowski hull derivatives."""
import unittest
from math import isclose, sqrt, tan, radians
from types import SimpleNamespace
import numpy as np
from openplaning.stability.derivatives import (
    TransverseCoefficientState, evaluate_hull_derivatives,
    coefficient_state_from_planing_boat, COEFFICIENT_UNITS,
)


def state(**change):
    params = dict(speed=15.0, rho=1025., g=9.8066, beam=4.15,
                  L_K=11.0, L_C=7.0, draft_transom=.5,
                  beta_deg=15., tau_deg=3., vcg=2.0, lcg=9.14)
    params.update(change)
    return TransverseCoefficientState(**params)


class CoefficientsTests(unittest.TestCase):
    def test_2004_key_coverage_and_finite(self):
        d = evaluate_hull_derivatives(state())
        self.assertEqual(set(d.values), set(COEFFICIENT_UNITS))
        self.assertTrue(all(np.isfinite(list(d.values.values()))))
        self.assertTrue(d.dimensionally_consistent)
        self.assertIs(d.require_dimensional_consistency(), d)

    def test_added_mass_symmetry(self):
        d = evaluate_hull_derivatives(state())
        self.assertEqual(d['Y_r_dot'], d['N_v_dot'])
        self.assertEqual(d['K_v_dot'], 0.)
        self.assertEqual(d['K_r_dot'], 0.)
        self.assertEqual(d['Y_phi_ddot'], 0.)

    def test_direct_2004_y_v_formula(self):
        s=state(); d=evaluate_hull_derivatives(s)
        expected=-.5*s.rho*s.speed*s.beam**2*(.6494*s.beta_deg**.6*s.t_ratio**2*s.cv**2)
        self.assertAlmostEqual(d['Y_v'], expected)

    def test_nr_1997_distinct_2004(self):
        s=state()
        r97=evaluate_hull_derivatives(s,source='lew1997_appendix')
        r04=evaluate_hull_derivatives(s,source='lew2004_table63')
        self.assertLess(r97['N_r'], 0)
        self.assertGreater(r04['N_r'], 0)
        self.assertNotEqual(r97['N_r'], r04['N_r'])

    def test_dimensional_1997_protection(self):
        d=evaluate_hull_derivatives(state(),source='lew1997_appendix')
        self.assertFalse(d.dimensionally_consistent)
        with self.assertRaisesRegex(ValueError, 'dimensionally unresolved'):
            d.require_dimensional_consistency()

    def test_published_97_vs_04_added_mass_equal(self):
        a=evaluate_hull_derivatives(state(),source='lew1997_appendix')
        b=evaluate_hull_derivatives(state(),source='lew2004_table63')
        for key in ('Y_v_dot','N_v_dot','N_r_dot','K_phi_ddot'):
            self.assertEqual(a[key],b[key])

    def test_dynroll_n_r_dot_has_2_instead_3(self):
        a=evaluate_hull_derivatives(state(),source='dynroll_prototype')
        b=evaluate_hull_derivatives(state(),source='lew2004_table63')
        self.assertNotEqual(a['N_r_dot'], b['N_r_dot'])
        self.assertFalse(a.dimensionally_consistent)

    def test_dynroll_missing_cv_square_in_yv(self):
        s=state(); a=evaluate_hull_derivatives(s,source='dynroll_prototype')
        b=evaluate_hull_derivatives(s,source='lew2004_table63')
        self.assertAlmostEqual(b['Y_v']/a['Y_v'], s.cv**2)

    def test_speed_scaling(self):
        x=state(speed=12); y=state(speed=24)
        a=evaluate_hull_derivatives(x); b=evaluate_hull_derivatives(y)
        self.assertAlmostEqual(b['Y_v']/a['Y_v'], 8.)  # U * Cv^2
        self.assertAlmostEqual(b['N_r']/a['N_r'], 2.)
        self.assertEqual(b['N_r_dot'],a['N_r_dot'])

    def test_rho_scaling(self):
        a=evaluate_hull_derivatives(state(rho=1000))
        b=evaluate_hull_derivatives(state(rho=2000))
        for k in COEFFICIENT_UNITS:
            self.assertAlmostEqual(b[k],2*a[k])

    def test_geometric_similarity_at_froude(self):
        a=state()
        q=2.3
        b=state(speed=a.speed*sqrt(q),beam=a.beam*q,L_K=a.L_K*q,
                L_C=a.L_C*q,draft_transom=a.draft_transom*q,
                vcg=a.vcg*q,lcg=a.lcg*q)
        x=evaluate_hull_derivatives(a);y=evaluate_hull_derivatives(b)
        # For every entry: dimensional powers [kg m^p / s^r], at constant Froude.
        expectations={'Y_v_dot':q**3,'K_v_dot':q**4,'N_v_dot':q**4,
            'Y_r_dot':q**4,'K_r_dot':q**5,'N_r_dot':q**5,
            'Y_phi_ddot':q**4,'K_phi_ddot':q**5,'N_phi_ddot':q**5,
            'Y_v':q**2.5,'K_v':q**3.5,'N_v':q**3.5,
            'Y_r':q**3.5,'K_r':q**4.5,'N_r':q**4.5,
            'Y_phi_dot':q**3.5,'K_phi_dot':q**4.5,'N_phi_dot':q**4.5}
        for name,ex in expectations.items():
            if abs(x[name])>1e-10:
                with self.subTest(name=name):
                    self.assertAlmostEqual(y[name]/x[name],ex,places=9)

    def test_1997_scaling_exposes_inconsistency(self):
        a=state();b=state(speed=a.speed*sqrt(2),beam=a.beam*2,
            L_K=a.L_K*2,L_C=a.L_C*2,draft_transom=a.draft_transom*2,
            vcg=a.vcg*2,lcg=a.lcg*2)
        p=evaluate_hull_derivatives(a,source='lew1997_appendix')
        q=evaluate_hull_derivatives(b,source='lew1997_appendix')
        # The reported mixed-dimension K_phi_dot is not expected to scale homogeneously.
        self.assertFalse(isclose(q['K_phi_dot'],p['K_phi_dot']*2**4.5,rel_tol=1e-6))

    def test_explicit_degree_convention(self):
        a=evaluate_hull_derivatives(state(beta_deg=15))
        b=evaluate_hull_derivatives(state(beta_deg=np.rad2deg(radians(15))))
        self.assertEqual(a['Y_v'],b['Y_v'])

    def test_input_failure_and_nans(self):
        for kwargs in (dict(speed=0),dict(beam=-1),dict(draft_transom=0),
                       dict(beta_deg=50),dict(L_C=12),dict(vcg=float('nan'))):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                state(**kwargs)
        with self.assertRaises(ValueError):evaluate_hull_derivatives(state(),source='2001')
        with self.assertRaises(TypeError):evaluate_hull_derivatives(None)

    def test_readonly_boat_extraction(self):
        s=state();boat=SimpleNamespace(speed=s.speed,rho=s.rho,g=s.g,beam=s.beam,
           L_K=s.L_K,L_C=s.L_C,T=s.draft_transom,beta=s.beta_deg,
           tau=s.tau_deg,vcg=s.vcg,lcg=s.lcg)
        prior=boat.__dict__.copy()
        self.assertEqual(coefficient_state_from_planing_boat(boat),s)
        self.assertEqual(boat.__dict__,prior)
        with self.assertRaisesRegex(ValueError,'missing'):
            coefficient_state_from_planing_boat(SimpleNamespace(speed=20))

    def test_h1_dimension_and_pitch_term(self):
        s=state(); d=evaluate_hull_derivatives(s)
        h1=(s.vcg-.3927*s.beam*tan(radians(s.beta_deg)))*(s.vcg-.306*s.beam)
        base=-.010237*s.rho*s.beam**5*s.lambda_w*(1-np.sin(radians(s.beta_deg)))
        self.assertAlmostEqual(d['K_phi_ddot'],base+h1*d['Y_v_dot'])


if __name__=='__main__':unittest.main()
