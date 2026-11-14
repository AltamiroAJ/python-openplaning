"""28C.10: source variants; literal regression values and no implicit default."""
import unittest
from math import radians, sin, cos, pi, tan, sqrt
from types import SimpleNamespace

import numpy as np
from openplaning.stability import (
    SourceFormulation, EvidenceLevel, SourceResolvedSystem,
    assemble_lewandowski_2004_tabulated,
    assemble_lewandowski_1996_full_projection,
    from_resolved_planing_boat, assert_28c7_reference,
    TransverseCoefficientState, evaluate_hull_derivatives,
    LewandowskiState, RollForceModel,
)
from openplaning.stability.source_audit import audit_roll_sources
from openplaning.stability.frame_resolved import assemble_lewandowski_2004_fsd
from openplaning.stability.transverse import RigidInertia


class SourceVariantTests(unittest.TestCase):
    def setUp(self):
        # Separate, explicit reproducible synthetic fixture, NOT the real 1600kg boat.
        self.hstate = TransverseCoefficientState(
            speed=12., rho=1025., g=9.81, beam=2.2,
            L_K=6.051, L_C=5., draft_transom=.28,
            beta_deg=18., tau_deg=3.0916, vcg=.65, lcg=2.25)
        self.rstate = LewandowskiState(
            speed=12., rho=1025., g=9.81, beam=2.2,
            L_K=6.051, L_C=5., L_C2=.2,
            beta_deg=18., tau_deg=3.0916, vcg=.65,
            lcg=2.25, lcp=2.6)
        self.hull=evaluate_hull_derivatives(self.hstate)
        self.roll=RollForceModel(self.rstate).derivatives()
        self.rigid=RigidInertia(1600., 4000., 6800., 144.)

    def tab(self):
        return assemble_lewandowski_2004_tabulated(self.hull,self.roll,
                       self.rigid,roll_state=self.rstate)

    def full(self):
        return assemble_lewandowski_1996_full_projection(self.hull,self.roll,
                       self.rigid,roll_state=self.rstate)

    def boat(self):
        s=self.rstate;h=self.hstate
        boat=SimpleNamespace(speed=h.speed,rho=h.rho,g=h.g,beam=h.beam,
          L_K=h.L_K,L_C=h.L_C,L_C2=s.L_C2,T=h.draft_transom,
          beta=h.beta_deg,tau=h.tau_deg,vcg=h.vcg,lcg=h.lcg,lcp=s.lcp,
          weight=1600*9.81,r_g=sqrt(4500/1600))
        snap=SimpleNamespace(mass=1600.,center_of_gravity=np.array([2.25,0.,.65]),
           inertia_tensor=np.array([[4000.,0,144.],[0,4500.,0],[144.,0,6800.]]))
        boat.get_mass_properties_snapshot=lambda:snap
        return boat

    def test_printed_source_y_literal(self):
        # Expected literal computed once from Table 6.2 expression at fixture,
        # not computed by calling the production function in the assertion.
        self.assertAlmostEqual(self.tab().heel_column[0],-8302.9813218,places=5)

    def test_printed_source_n_literal(self):
        # Direct algebra: -8302.9813218 * (2.6 - 2.25) = -2906.04346263.
        self.assertAlmostEqual(self.tab().heel_column[2],-2906.04346263,places=5)

    def test_printed_source_k_literal(self):
        self.assertAlmostEqual(self.tab().heel_column[1],-38364.34562272,places=5)

    def test_full_projection_y_literal_in_fsd(self):
        # 1996 FPU force rate +6943.50720082, sign flips to FSD.
        self.assertAlmostEqual(self.full().heel_column[0],-6943.50720082,places=4)

    def test_2004_y_table_manual_expression(self):
        s=self.rstate
        b=s.beam;beta=radians(s.beta_deg);tau=radians(s.tau_deg)
        lam=(s.L_K+s.L_C)/(2*b)
        # Deliberate independent expression mirroring printed equation.
        fphi=-.5*s.rho*s.speed**2*b**2/(4*cos(beta)**2)*(
            ((1-sin(beta))*pi*lam/(1+lam)+1.33/pi)*tau +
            (1-sin(beta))/(cos(beta)*2*(1+lam)**2))
        self.assertAlmostEqual(self.tab().heel_column[0],2*fphi*sin(beta),places=5)

    def test_n_is_y_times_lever_at_varying_lcp(self):
        from dataclasses import replace
        for arm in (0.,-.3,.2,.6):
            r=replace(self.rstate,lcp=self.rstate.lcg+arm)
            a=assemble_lewandowski_2004_tabulated(self.hull,self.roll,self.rigid,roll_state=r)
            self.assertAlmostEqual(a.heel_column[2],a.heel_column[0]*arm,places=8)

    def test_independent_28c7_source_audit_cross_check(self):
        a=audit_roll_sources(self.rstate)
        tab=self.tab();full=self.full()
        self.assertAlmostEqual(tab.heel_column[0],a.y_phi_2004_table63_N_per_rad,places=7)
        self.assertAlmostEqual(tab.heel_column[2],a.n_phi_2004_table63_Nm_per_rad,places=7)
        self.assertAlmostEqual(full.heel_column[0],-a.y_phi_full_projection_N_per_rad,places=5)

    def test_tabulated_eigenvalue_literal_regression(self):
        # Independent fixed synthetic fixture baseline, for numerical regression.
        got=self.tab().eigenvalues
        for wanted in (-2.70912199+1.11911401j,-2.70912199-1.11911401j,
                       -.8573793497+2.28517413j,-.8573793497-2.28517413j):
            self.assertTrue(any(abs(z-wanted)<1e-6 for z in got),wanted)
        self.assertAlmostEqual(self.tab().spectral_abscissa,-.8573793496646805,places=7)

    def test_full_projection_eigenvalue_literal_regression(self):
        got=self.full().eigenvalues
        for wanted in (-2.67276274+.93120465j,-2.67276274-.93120465j,
                       -.8937385974+2.37490996j,-.8937385974-2.37490996j):
            self.assertTrue(any(abs(z-wanted)<1e-6 for z in got),wanted)

    def test_full_projection_exactly_equals_28c5(self):
        a=self.full().linear_system
        b=assemble_lewandowski_2004_fsd(self.hull,self.roll,self.rigid,roll_state=self.rstate)
        np.testing.assert_allclose(a.state_matrix,b.state_matrix,rtol=0,atol=0)
        np.testing.assert_allclose(a.heel_column,b.heel_column,rtol=0,atol=0)

    def test_other_matrices_unchanged_between_variants(self):
        a=self.tab();b=self.full()
        np.testing.assert_array_equal(a.linear_system.acceleration_matrix,
                                      b.linear_system.acceleration_matrix)
        np.testing.assert_array_equal(a.linear_system.rate_matrix,
                                      b.linear_system.rate_matrix)
        self.assertNotAlmostEqual(a.heel_column[0],b.heel_column[0],places=3)

    def test_evidence_enum_is_mandatory(self):
        a=self.tab();b=self.full()
        self.assertIs(a.evidence_level,EvidenceLevel.DOCUMENTAL_FIDELITY)
        self.assertIs(b.evidence_level,EvidenceLevel.DIAGNOSTIC)
        self.assertIsInstance(a.formulation,SourceFormulation)
        with self.assertRaises(TypeError):
            SourceResolvedSystem(a.formulation,'not_an_enum',a.linear_system,
                                 a.reference,a.frame,a.units,a.limitations)

    def test_public_facade_has_no_default(self):
        with self.assertRaises(TypeError):from_resolved_planing_boat(self.boat())

    def test_public_facade_rejects_free_text(self):
        with self.assertRaises(TypeError):
            from_resolved_planing_boat(self.boat(),formulation='lewandowski_2004_tabulated')

    def test_explicit_selectors(self):
        boat=self.boat()
        a=from_resolved_planing_boat(boat,formulation=SourceFormulation.LEWANDOWSKI_2004_TABULATED)
        b=from_resolved_planing_boat(boat,formulation=SourceFormulation.LEWANDOWSKI_1996_FULL_PROJECTION)
        self.assertEqual(a.evidence_level,EvidenceLevel.DOCUMENTAL_FIDELITY)
        self.assertEqual(b.evidence_level,EvidenceLevel.DIAGNOSTIC)

    def test_facade_no_mutation(self):
        boat=self.boat()
        before={k:v for k,v in vars(boat).items() if k!='get_mass_properties_snapshot'}
        from_resolved_planing_boat(boat,formulation=SourceFormulation.LEWANDOWSKI_2004_TABULATED)
        self.assertEqual(before,{k:v for k,v in vars(boat).items() if k!='get_mass_properties_snapshot'})

    def test_reference_assertion_does_not_accept_wrong_fixture(self):
        # R5 literal targets are for the user's real resolved boat, not this fixture.
        with self.assertRaises(AssertionError):assert_28c7_reference(self.boat())

    def test_invalid_hull_source_rejected(self):
        other=evaluate_hull_derivatives(self.hstate,source='lew1997_appendix')
        with self.assertRaises(ValueError):
            assemble_lewandowski_2004_tabulated(other,self.roll,self.rigid,roll_state=self.rstate)

    def test_invalid_roll_source_rejected(self):
        other=RollForceModel(self.rstate,source_variant='dynroll_1997_prototype').derivatives()
        with self.assertRaises(ValueError):
            assemble_lewandowski_1996_full_projection(self.hull,other,self.rigid,roll_state=self.rstate)

    def test_source_label_and_limitations_are_explicit(self):
        a=self.tab();b=self.full()
        for x in (a,b):
            self.assertIn('Lewandowski',x.reference)
            self.assertIn('FSD',x.frame)
            self.assertTrue(x.limitations)
            self.assertTrue(any('validat' in s.lower() or 'validated' in s.lower() for s in x.limitations))


if __name__ == '__main__':unittest.main()
