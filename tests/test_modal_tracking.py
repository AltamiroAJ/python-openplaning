"""28C.6: branch continuation, modal participation, dominance crossing."""
import unittest
from math import sqrt
from types import SimpleNamespace
import numpy as np

from openplaning.stability.modal_tracking import (
    track_oscillatory_modes, sweep_ixz_fsd,
)
from openplaning.stability.transverse import linearize_transverse_system


def analytic_system(xz, *, coupling=0.):
    """Two damped oscillator pairs; derivative of phi equals p."""
    droll=2.-xz
    dsway=2.+xz
    A=np.zeros((4,4))
    A[1,2]=1.
    A[2,1]=-(droll*droll+4.)
    A[2,2]=-2*droll
    A[0,0]=A[3,3]=-dsway
    A[0,3]=4.
    A[3,0]=-4.
    A[0,2]=coupling
    M=-np.diag([10.,20.,30.])
    Q=-M@A[[0,2,3],:]
    D=Q[:,[0,2,3]]
    h=Q[:,1]
    return linearize_transverse_system(M,D,h,mass=10,beam=2)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.grid=np.linspace(-0.9,0.9,91)
        self.result=track_oscillatory_modes(self.grid,analytic_system,
                                            speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)

    def test_synthetic_branch_analytic_values(self):
        o=self.result
        # First branch is roll oscillator, lower frequency, throughout sweep.
        np.testing.assert_allclose(o.eigenvalues[:,0].real,-(2-self.grid),atol=1e-12)
        np.testing.assert_allclose(o.eigenvalues[:,0].imag,2.,atol=1e-12)
        np.testing.assert_allclose(o.eigenvalues[:,1].real,-(2+self.grid),atol=1e-12)
        np.testing.assert_allclose(o.eigenvalues[:,1].imag,4.,atol=1e-12)

    def test_dominance_switch(self):
        o=self.result
        self.assertEqual(len(o.dominance_brackets),1)
        left,right,estimate=o.dominance_brackets[0]
        self.assertLessEqual(left,0)
        self.assertGreaterEqual(right,0)
        self.assertAlmostEqual(estimate,0.,places=10)
        self.assertEqual(o.dominant_branch[0],1)
        self.assertEqual(o.dominant_branch[-1],0)
        np.testing.assert_allclose(o.spectral_abscissa,
                                   np.maximum(o.eigenvalues[:,0].real,o.eigenvalues[:,1].real))

    def test_shapes_are_scaled_normalized_and_phased(self):
        norms=np.linalg.norm(self.result.dimensionless_shapes,axis=2)
        np.testing.assert_allclose(norms,1.,atol=1e-13)
        for i in range(1,len(self.grid)):
            for branch in range(2):
                ov=np.vdot(self.result.dimensionless_shapes[i-1,branch],
                           self.result.dimensionless_shapes[i,branch])
                self.assertGreaterEqual(ov.real,-1e-12)
                self.assertAlmostEqual(ov.imag,0.,places=9)

    def test_kinetic_fractions_and_roll_shape(self):
        f=self.result.kinetic_fractions
        np.testing.assert_allclose(f.sum(axis=2),1.,atol=1e-13)
        self.assertTrue(np.all(f[:,0,1]>.99))  # roll oscillator
        self.assertTrue(np.all(f[:,1,1]<1e-12))
        self.assertTrue(np.all(self.result.kinetic_labels[:,0]=='roll_rate'))

    def test_immutable_outputs(self):
        with self.assertRaises(ValueError):
            self.result.eigenvalues[0,0]=1.

    def test_csv_rows_are_complete(self):
        rows=self.result.as_rows()
        self.assertEqual(len(rows),2*len(self.grid))
        self.assertEqual(sum(int(r['dominant']) for r in rows),len(self.grid))
        self.assertIn('assignment_margin',rows[0])
        self.assertIn('roll_kinetic_fraction',rows[0])

    def test_invalid_grid(self):
        for values in ([0,0,1],[0,1],[0,np.nan,2],[1,0,-1]):
            with self.assertRaises(ValueError):
                track_oscillatory_modes(values,analytic_system,
                    speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)

    def test_insufficient_complex_pairs(self):
        def factory(x):
            # A case with all-real eigenvalues (overdamped roll and yaw).
            M=np.diag([-10.,-20.,-30.])
            D=np.diag([-20.,-6.,-15.]);h=np.array([0.,0.,0.])
            return linearize_transverse_system(M,D,h,mass=10,beam=2)
        with self.assertRaisesRegex(ValueError,'two distinct underdamped'):
            track_oscillatory_modes([0,1,2],factory,
                speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)

    def test_wrong_factory_type(self):
        with self.assertRaises(TypeError):
            track_oscillatory_modes([0,1,2],lambda x:np.eye(4),
                speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)

    def test_assignment_margin_and_conditions(self):
        self.assertTrue(np.all(np.isfinite(self.result.eigenvector_condition)))
        self.assertTrue(np.all((self.result.assignment_margin>=0)&(self.result.assignment_margin<=1+1e-12)))
        self.assertTrue(np.all(self.result.complex_separation>0))

    def test_crossing_not_spectral_degeneracy(self):
        # Equal real parts at x=0, yet distinct nonzero frequencies.
        k=np.argmin(abs(self.grid))
        self.assertAlmostEqual(self.result.eigenvalues[k,0].real,
                               self.result.eigenvalues[k,1].real,places=12)
        self.assertGreater(self.result.complex_separation[k],1.)

    def test_coupled_case_still_has_finite_tracking(self):
        result=track_oscillatory_modes(np.linspace(-.4,.4,41),
            lambda x:analytic_system(x,coupling=.07),
            speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)
        self.assertTrue(np.all(np.isfinite(result.eigenvalues)))
        self.assertTrue(np.all(np.isfinite(result.kinetic_fractions)))


class FullInertiaSweepTests(unittest.TestCase):
    def _boat(self):
        model=SimpleNamespace(
            speed=12.,rho=1025.,g=9.81,beam=2.2,L_K=6.05,L_C=5.,L_C2=0.,
            T=.28,beta=18.,tau=3.0916,vcg=.65,lcg=2.25,lcp=2.6,
            weight=1600*9.81,r_g=sqrt(5000/1600))
        snap=SimpleNamespace(mass=1600.,center_of_gravity=np.array([2.25,0.,.65]),
           inertia_tensor=np.array([[4000.,0.,0.],[0.,5000.,0.],[0.,0.,6800.]]))
        model.get_mass_properties_snapshot=lambda:snap
        return model

    def test_fsd_sweep_no_mutation(self):
        boat=self._boat()
        base=boat.get_mass_properties_snapshot().inertia_tensor.copy()
        values=np.linspace(-144.,144.,41)
        o=sweep_ixz_fsd(boat,values)
        np.testing.assert_array_equal(o.ixz_tensor_fpu,values)
        np.testing.assert_array_equal(boat.get_mass_properties_snapshot().inertia_tensor,base)
        self.assertEqual(o.eigenvalues.shape,(len(values),2))
        self.assertTrue(np.all(np.isfinite(o.eigenvalues)))

    def test_invalid_tensor_at_extreme(self):
        with self.assertRaisesRegex(ValueError,'nonphysical'):
            sweep_ixz_fsd(self._boat(),[-7000.,0.,7000.])

    def test_sweep_does_not_solve_equilibrium(self):
        boat=self._boat()
        self.assertFalse(hasattr(boat,'get_steady_trim'))
        self.assertEqual(len(sweep_ixz_fsd(boat,[-100.,0.,100.]).eigenvalues),3)

    def test_orthogonal_axes_survive_limited_sweep(self):
        obj=sweep_ixz_fsd(self._boat(),[-144.,0.,144.])
        np.testing.assert_allclose(obj.kinetic_fractions.sum(axis=2),1,atol=1e-12)


if __name__ == '__main__':unittest.main()

class RefinementTests(unittest.TestCase):
    def test_analytic_crossing_refinement(self):
        from openplaning.stability.modal_tracking import refine_dominance_crossings
        grid=np.linspace(-.91,.89,21)
        sweep=track_oscillatory_modes(grid,analytic_system,
            speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)
        found=refine_dominance_crossings(sweep,analytic_system,speed=10.,beam=2.)
        self.assertEqual(len(found),1)
        self.assertAlmostEqual(found[0].ixz_tensor_fpu,0.,delta=1e-7)
        self.assertGreater(found[0].complex_gap_per_second,1.)
        self.assertGreater(found[0].minimum_assignment_margin,0.)

    def test_empty_crossings(self):
        from openplaning.stability.modal_tracking import refine_dominance_crossings
        sweep=track_oscillatory_modes([.1,.2,.3],analytic_system,
            speed=10.,beam=2.,mass=10.,ixx=20.,izz=30.)
        self.assertEqual(refine_dominance_crossings(sweep,analytic_system,speed=10.,beam=2.),())

class PublicAnalysisTests(unittest.TestCase):
    def test_one_call_returns_sweep_and_refined_crossings(self):
        from openplaning.stability.modal_tracking import analyze_ixz_fsd
        boat=FullInertiaSweepTests()._boat()
        sweep,crossings=analyze_ixz_fsd(boat,np.linspace(-144,144,41))
        self.assertEqual(len(sweep.ixz_tensor_fpu),41)
        self.assertIsInstance(crossings,tuple)
        self.assertTrue(np.all(np.isfinite(sweep.spectral_abscissa)))
