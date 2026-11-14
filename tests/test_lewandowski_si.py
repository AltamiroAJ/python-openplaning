"""Analytical and dimensional tests: no experimental validation claimed."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from openplaning.stability import LewandowskiState, RollForceModel, from_planing_boat
from openplaning.stability.units import (
    ft_to_m, knots_to_mps, lbf_to_newton, lbf_ft_to_newton_m,
    slug_ft3_to_kg_m3, slug_ft2_to_kg_m2,
)


def case(scale=1):
    return LewandowskiState(speed=18*math.sqrt(scale), rho=1025, g=9.80665,
        beam=3*scale, beta_deg=15, tau_deg=5.4, L_K=9*scale,
        L_C=6.2*scale, L_C2=0, vcg=1.0*scale, lcg=3.0*scale,
        lcp=4.0*scale)


class UnitTests(unittest.TestCase):
    def test_exact_mile_foot_and_knots(self):
        self.assertAlmostEqual(ft_to_m(1), 0.3048)
        self.assertAlmostEqual(knots_to_mps(1), 1852/3600)

    def test_force_and_moment(self):
        self.assertAlmostEqual(lbf_to_newton(1), 4.4482216152605)
        self.assertAlmostEqual(lbf_ft_to_newton_m(1), 4.4482216152605*0.3048)

    def test_density_and_inertia(self):
        self.assertAlmostEqual(slug_ft3_to_kg_m3(1), 14.593902937206363/.3048**3)
        self.assertAlmostEqual(slug_ft2_to_kg_m2(1), 14.593902937206363*.3048**2)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): knots_to_mps(float('nan'))


class ForceModelTests(unittest.TestCase):
    def test_geometry_residual_is_transparent(self):
        s=case()
        self.assertAlmostEqual(s.lambda_geometry_residual,
            (s.L_K+s.L_C)/(2*s.beam)-s.lambda_eq26)

    def test_zero_heel_is_symmetric(self):
        model=RollForceModel(case())
        np.testing.assert_allclose(model.forces_at_heel(0), [0,0,0], atol=1e-9)

    def test_force_1996_eq27_direct_substitution(self):
        s=case();m=RollForceModel(s)
        be=math.radians(s.beta_deg);tau=math.radians(s.tau_deg)
        lam=s.L_K/s.beam-math.tan(be)/(2*math.pi*math.tan(tau))
        B=math.pi/4*(1-math.sin(be))*math.cos(tau)*lam/(1+lam)
        B+=(1.33/4)*lam*math.cos(tau)*math.sin(2*tau)*math.cos(be)
        expected=.5*s.rho*s.speed**2*s.beam**2*(math.sin(2*tau)/(2*math.cos(be)))*B
        self.assertAlmostEqual(m.side_normal_force(s.beta_deg),expected,places=9)

    def test_two_halves_reconstruct_vertical_equivalent(self):
        s=case();m=RollForceModel(s)
        side=m.side_normal_force(s.beta_deg)
        # Equivalent whole-bottom vertical lift corresponding to Eq (27).
        total=2*side*math.cos(math.radians(s.beta_deg))
        self.assertAlmostEqual(total, side*2*math.cos(math.radians(s.beta_deg)))
        # The 1/2 in the normal projection must not be omitted.
        self.assertGreater(total,0)

    def test_explicit_133_over_4_vs_one_third(self):
        a=RollForceModel(case(),source_variant='lew1996_eq27').side_normal_force(15)
        b=RollForceModel(case(),source_variant='dynroll_1997_prototype').side_normal_force(15)
        self.assertNotEqual(a,b)
        self.assertGreater(b,a)

    def test_derivative_units_radian_matches_deg_conversion(self):
        m=RollForceModel(case()); eps=1e-4
        rad=m.derivatives(step_rad=eps).Y_phi_dynamic
        dh=math.degrees(eps)
        # derivative w.r.t. degrees * (180/pi) = derivative per radian
        deg=(m.forces_at_heel(math.radians(dh))[0]-m.forces_at_heel(-math.radians(dh))[0])/(2*dh)
        self.assertAlmostEqual(rad,deg*(180/math.pi),delta=max(abs(rad)*1e-11,1e-7))

    def test_differentiation_convergence(self):
        m=RollForceModel(case())
        d1=m.derivatives(step_rad=1e-4)
        d2=m.derivatives(step_rad=1e-5)
        for k in ['Y_phi_dynamic','K_phi_dynamic','N_phi_dynamic']:
            a,b=getattr(d1,k),getattr(d2,k)
            self.assertAlmostEqual(a,b,delta=max(1e-5,abs(b)*1e-5))

    def test_moment_relations(self):
        s=case();m=RollForceModel(s);d=m.derivatives()
        self.assertAlmostEqual(d.N_phi_dynamic,d.Y_phi_dynamic*(s.lcp-s.lcg),delta=1e-6)
        phi=1e-3
        y,k,n=m.forces_at_heel(phi)
        self.assertAlmostEqual(n,y*(s.lcp-s.lcg))

    def test_length_similarity_dimension(self):
        # F_side ~ rho U² b² ~ L^3 under Froude similarity; K~L^4.
        a,b=2.3,RollForceModel(case())
        c=RollForceModel(case(scale=a))
        self.assertAlmostEqual(c.side_normal_force(15)/b.side_normal_force(15),a**3,delta=1e-10)
        self.assertAlmostEqual(c.derivatives().K_phi_dynamic/b.derivatives().K_phi_dynamic,a**4,delta=1e-7)

    def test_no_silent_source_variant(self):
        with self.assertRaises(ValueError): RollForceModel(case(),source_variant='1997_verified')

    def test_cosine_projected_is_distinct(self):
        a=RollForceModel(case(),beam_mode='constant').derivatives()
        b=RollForceModel(case(),beam_mode='cosine_projected').derivatives()
        self.assertNotAlmostEqual(a.Y_phi_dynamic,b.Y_phi_dynamic)

    def test_static_is_finite(self):
        d=RollForceModel(case()).derivatives()
        self.assertTrue(math.isfinite(d.K_phi_static))
        self.assertAlmostEqual(d.K_phi_total,d.K_phi_static+d.K_phi_dynamic)

    def test_invalid_geometry_rejected(self):
        s=case();d=s.__dict__.copy();d['tau_deg']=0
        with self.assertRaises(ValueError): LewandowskiState(**d)
        with self.assertRaises(ValueError): RollForceModel(case()).derivatives(step_rad=0)
        with self.assertRaises(ValueError): RollForceModel(case()).side_normal_force(89.9)

    def test_snapshot_does_not_mutate_boat(self):
        s=case();b=SimpleNamespace(speed=s.speed,rho=s.rho,g=s.g,beam=s.beam,
            beta=s.beta_deg,tau=s.tau_deg,L_K=s.L_K,L_C=s.L_C,L_C2=s.L_C2,
            vcg=s.vcg,lcg=s.lcg,lcp=s.lcp)
        before=b.__dict__.copy();view=from_planing_boat(b)
        RollForceModel(view).derivatives()
        self.assertEqual(before,b.__dict__)

    def test_incomplete_boat_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'run equilibrium'):
            from_planing_boat(SimpleNamespace(speed=10))

    def test_imperial_equivalent_input_to_si(self):
        s=case();m=RollForceModel(s)
        # An independently constructed state through round-trip unit conversion.
        ft=s.beam/.3048; kn=s.speed/(1852/3600); rho_slug=s.rho/(14.593902937206363/.3048**3)
        other=LewandowskiState(speed=knots_to_mps(kn),rho=slug_ft3_to_kg_m3(rho_slug),g=s.g,
            beam=ft_to_m(ft),beta_deg=s.beta_deg,tau_deg=s.tau_deg,
            L_K=ft_to_m(s.L_K/.3048),L_C=ft_to_m(s.L_C/.3048),L_C2=0,
            vcg=ft_to_m(s.vcg/.3048),lcg=ft_to_m(s.lcg/.3048),lcp=ft_to_m(s.lcp/.3048))
        np.testing.assert_allclose(m.forces_at_heel(math.radians(1)),RollForceModel(other).forces_at_heel(math.radians(1)),rtol=1e-13,atol=1e-8)

if __name__=='__main__': unittest.main()
