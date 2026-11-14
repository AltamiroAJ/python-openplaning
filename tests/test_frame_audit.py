"""28C.4 independent physical frame audit, no change to 28C.3 model."""
import unittest
import numpy as np
from openplaning.stability.frame_audit import (
    R_FPU_TO_FSD,T_Q_FPU_TO_FSD,T_W_FPU_TO_FSD,T_X_FPU_TO_FSD,
    DescriptorTransform,book_xz_raw_from_fpu_tensor,
    fpu_xz_tensor_from_book_raw,transform_spatial_quantities,
)
from openplaning.stability.transverse import linearize_transverse_system, RigidInertia


class FrameAudit(unittest.TestCase):
    def setUp(self):
        self.M=np.array([[-1600.,13.,-25.],[13.,-2500.,144.],[-25.,144.,-3900.]])
        self.D=np.array([[-800.,15.,300.],[50.,-200.,-32.],[40.,-65.,-450.]])
        self.h=np.array([12.,-600.,-50.])
        self.transform=DescriptorTransform.physical_fpu_to_fsd()

    def test_spatial_proper_rotation(self):
        R=R_FPU_TO_FSD
        np.testing.assert_allclose(R.T@R,np.eye(3));self.assertAlmostEqual(np.linalg.det(R),1.)

    def test_polar_vector_and_axial_moment(self):
        o=transform_spatial_quantities(R_FPU_TO_FSD,vector=[1,2,3],moment=[4,5,6])
        np.testing.assert_array_equal(o['vector'],[1,-2,-3])
        np.testing.assert_array_equal(o['moment'],[4,-5,-6])

    def test_physical_inertia_sign_and_trace(self):
        I=np.array([[3000.,0.,144.],[0.,3400.,0.],[144.,0.,5000.]])
        J=transform_spatial_quantities(R_FPU_TO_FSD,inertia=I)['inertia']
        self.assertEqual(J[0,2],-144.)
        self.assertEqual(book_xz_raw_from_fpu_tensor(I),144.)
        self.assertEqual(fpu_xz_tensor_from_book_raw(144.),144.)
        np.testing.assert_allclose(np.linalg.eigvalsh(I),np.linalg.eigvalsh(J))
        self.assertEqual(np.trace(I),np.trace(J))

    def test_old_28c3_property_has_opposite_sign_for_book_axes(self):
        # This test documents a discrepancy; it does NOT patch the old solver.
        rigid=RigidInertia(1600,3000,5000,144.)
        tensor=np.array([[3000.,0.,144.],[0.,3400.,0.],[144.,0.,5000.]])
        self.assertEqual(rigid.Ixz_book_raw,-144.)
        self.assertEqual(book_xz_raw_from_fpu_tensor(tensor),144.)

    def test_inertia_two_signs(self):
        for s in [-144.,144.]:
            I=np.array([[3000.,0.,s],[0.,3400.,0.],[s,0.,5000.]])
            self.assertEqual(book_xz_raw_from_fpu_tensor(I),s)

    def test_descriptor_signs_by_chain_rule(self):
        M,D,h=self.transform.descriptor(self.M,self.D,self.h)
        T=T_Q_FPU_TO_FSD
        np.testing.assert_allclose(M,T@self.M@T)
        np.testing.assert_allclose(D,T@self.D@T)
        np.testing.assert_allclose(h,T@self.h)
        self.assertEqual(D[0,1],-self.D[0,1])
        self.assertEqual(D[1,2],-self.D[1,2])
        self.assertEqual(h[0],-self.h[0])
        self.assertEqual(h[1],self.h[1])

    def test_all_derivative_signs_explicitly(self):
        # For FPU -> FSD, derivatives d[Y,K,N]/d[v,p,r].
        signs=np.outer([-1,1,-1],[-1,1,-1])
        expected=np.array([[1,-1,1],[-1,1,-1],[1,-1,1]])
        np.testing.assert_array_equal(signs,expected)
        mat=np.arange(1,10.,dtype=float).reshape(3,3)
        converted=self.transform.derivative_blocks(mat,mat,np.array([1.,2.,3.]))
        np.testing.assert_allclose(converted[0],signs*mat)
        np.testing.assert_allclose(converted[1],signs*mat)
        np.testing.assert_allclose(converted[2],[-1,2,-3])

    def test_generalized_power_is_invariant_under_physical_frame(self):
        q=np.array([2.,0.4,-0.2])
        w=np.array([150.,300.,-200.])
        qb=self.transform.q@q
        wb=self.transform.wrench@w
        self.assertAlmostEqual(float(q@w),float(qb@wb))

    def test_kinematic_roll_preservation(self):
        np.testing.assert_allclose(self.transform.state,T_X_FPU_TO_FSD)
        self.assertEqual(self.transform.state[1,1],self.transform.state[2,2])

    def test_eigenvalues_invariant_under_real_physical_180deg_frame(self):
        a=linearize_transverse_system(self.M,self.D,self.h,mass=1600,beam=2.2)
        Mb,Db,hb=self.transform.descriptor(self.M,self.D,self.h)
        b=linearize_transverse_system(Mb,Db,hb,mass=1600,beam=2.2)
        np.testing.assert_allclose(self.transform.state_similarity(a.state_matrix),b.state_matrix,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.sort_complex(a.eigenvalues),np.sort_complex(b.eigenvalues),atol=1e-11)
        np.testing.assert_allclose(a.characteristic_polynomial/a.characteristic_polynomial[0],b.characteristic_polynomial/b.characteristic_polynomial[0],rtol=1e-12,atol=1e-12)

    def test_double_transform_returns_original_descriptor(self):
        M,D,h=self.transform.descriptor(self.M,self.D,self.h)
        MM,DD,hh=self.transform.descriptor(M,D,h)
        np.testing.assert_allclose(MM,self.M);np.testing.assert_allclose(DD,self.D);np.testing.assert_allclose(hh,self.h)

    def test_scaling_admissible_if_p_follows_phi(self):
        T=DescriptorTransform(np.diag([2.,-3.,4.]),np.diag([1.,-1.,1.]),-3.)
        M,D,h=T.descriptor(self.M,self.D,self.h)
        a=linearize_transverse_system(self.M,self.D,self.h,mass=1600,beam=2.2)
        b=linearize_transverse_system(M,D,h,mass=1600,beam=2.2)
        np.testing.assert_allclose(T.state_similarity(a.state_matrix),b.state_matrix,atol=1e-11)

    def test_invalid_permutation_that_breaks_phi_kinematics_rejected(self):
        with self.assertRaisesRegex(ValueError,'p coordinate'):
            DescriptorTransform(np.array([[0,1,0],[1,0,0],[0,0,1]]),np.eye(3),1.)

    def test_reject_reflection_and_anisotropic_scaling_as_spatial_rotations(self):
        for r in [np.diag([-1,1,1]),np.diag([1,2,1])]:
            with self.assertRaises(ValueError):transform_spatial_quantities(r,vector=[1,2,3])

    def test_reject_bad_descriptor_shape(self):
        with self.assertRaises(ValueError):self.transform.descriptor(np.eye(2),self.D,self.h)

    def test_do_not_treat_inertia_sign_flip_as_coordinate_transform(self):
        M2=self.M.copy();M2[1,2]*=-1;M2[2,1]*=-1
        a=linearize_transverse_system(self.M,self.D,self.h,mass=1600,beam=2.2)
        b=linearize_transverse_system(M2,self.D,self.h,mass=1600,beam=2.2)
        self.assertGreater(np.max(np.abs(np.sort_complex(a.eigenvalues)-np.sort_complex(b.eigenvalues))),1e-5)

    def test_invalid_axis_is_not_a_reduced_3dof_frame_change(self):
        # Full spatial rotation is permitted but DOES NOT automatically map
        # the restricted sway-roll-yaw descriptor to itself.
        theta=np.pi/2
        R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
        out=transform_spatial_quantities(R,vector=[0,1,0])
        np.testing.assert_allclose(out['vector'],[-1,0,0],atol=1e-15)
        # A surge component is now required; no direct 3-DOF equivalence.

if __name__=='__main__':unittest.main()
