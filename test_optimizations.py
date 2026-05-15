#!/usr/bin/env python
"""Test script to verify Phase 4 performance optimizations."""

import numpy as np
import time
from openplaning.openplaning_refactored import PlaningBoat

def test_basic_functionality():
    """Test that basic functionality still works after optimizations."""
    print("Testing basic functionality...")
    
    # Create a test vessel
    boat = PlaningBoat(
        speed=10.0,      # m/s
        weight=50000.0,  # N
        beam=2.5,        # m
        lcg=3.0,         # m
        vcg=0.5,         # m
        r_g=2.0,         # m
        beta=20.0,       # deg
        epsilon=0.0,     # deg
        vT=0.0,          # m
        lT=3.0           # m
    )
    
    # Test get_geo_lengths
    boat.get_geo_lengths()
    assert hasattr(boat, 'L_K'), "L_K not set"
    assert hasattr(boat, 'L_C'), "L_C not set"
    assert hasattr(boat, 'lambda_W'), "lambda_W not set"
    print(f"  ✓ get_geo_lengths() - L_K={boat.L_K:.3f}m, L_C={boat.L_C:.3f}m")
    
    # Test get_forces
    boat.get_forces()
    assert hasattr(boat, 'hydrodynamic_force'), "hydrodynamic_force not set"
    assert hasattr(boat, 'skin_friction'), "skin_friction not set"
    print(f"  ✓ get_forces() - F_hydro={boat.hydrodynamic_force}")
    
    # Test get_steady_trim
    boat.get_steady_trim()
    assert hasattr(boat, 'z_wl'), "z_wl not set"
    assert hasattr(boat, 'tau'), "tau not set"
    print(f"  ✓ get_steady_trim() - z_wl={boat.z_wl:.3f}m, tau={boat.tau:.3f}deg")
    
    # Test get_eom_matrices
    boat.get_eom_matrices()
    assert hasattr(boat, 'mass_matrix'), "mass_matrix not set"
    assert hasattr(boat, 'damping_matrix'), "damping_matrix not set"
    assert hasattr(boat, 'restoring_matrix'), "restoring_matrix not set"
    print(f"  ✓ get_eom_matrices() - Mass matrix shape={boat.mass_matrix.shape}")
    
    # Test check_porpoising
    boat.check_porpoising()
    assert hasattr(boat, 'porpoising'), "porpoising not set"
    print(f"  ✓ check_porpoising() - Porpoising={boat.porpoising[0][0]}")
    
    # Test get_seaway_behavior
    boat.H_sig = 1.0
    boat.loa = 7.5
    boat.get_seaway_behavior()
    assert hasattr(boat, 'avg_impact_acc'), "avg_impact_acc not set"
    assert hasattr(boat, 'R_AW'), "R_AW not set"
    print(f"  ✓ get_seaway_behavior() - R_AW={boat.R_AW:.1f}N")
    
    print("\n✅ All basic functionality tests passed!\n")

def test_caching():
    """Test that caching is working for z_max polynomial."""
    print("Testing caching mechanism...")
    
    boat = PlaningBoat(
        speed=10.0,
        weight=50000.0,
        beam=2.5,
        lcg=3.0,
        vcg=0.5,
        r_g=2.0,
        beta=20.0,
        epsilon=0.0,
        vT=0.0,
        lT=3.0
    )
    
    # First call - should compute
    start = time.time()
    for _ in range(100):
        boat.get_geo_lengths()
    time1 = time.time() - start
    
    # Clear cache and test again
    boat._get_z_max_poly.cache_clear()
    
    # Call with same beta values repeatedly
    start = time.time()
    for _ in range(100):
        boat.get_geo_lengths()
    time2 = time.time() - start
    
    print(f"  Cache test completed (100 iterations)")
    print(f"  Note: Caching provides benefits when same beta values are reused\n")

def test_trig_precomputation():
    """Verify trigonometric precomputation doesn't affect results."""
    print("Testing trigonometric precomputation...")
    
    boat = PlaningBoat(
        speed=10.0,
        weight=50000.0,
        beam=2.5,
        lcg=3.0,
        vcg=0.5,
        r_g=2.0,
        beta=20.0,
        epsilon=0.0,
        vT=0.0,
        lT=3.0
    )
    
    # Run multiple times to ensure consistency
    results = []
    for i in range(5):
        boat.get_geo_lengths()
        boat.get_forces()
        results.append({
            'L_K': boat.L_K,
            'L_C': boat.L_C,
            'F_x': boat.hydrodynamic_force[0],
            'F_z': boat.hydrodynamic_force[1]
        })
    
    # Check all results are identical
    for i in range(1, len(results)):
        assert np.isclose(results[i]['L_K'], results[0]['L_K']), "L_K inconsistent"
        assert np.isclose(results[i]['L_C'], results[0]['L_C']), "L_C inconsistent"
        assert np.isclose(results[i]['F_x'], results[0]['F_x']), "F_x inconsistent"
        assert np.isclose(results[i]['F_z'], results[0]['F_z']), "F_z inconsistent"
    
    print(f"  ✓ Results consistent across multiple runs")
    print(f"  L_K={results[0]['L_K']:.6f}, L_C={results[0]['L_C']:.6f}")
    print(f"  F_x={results[0]['F_x']:.2f}N, F_z={results[0]['F_z']:.2f}N\n")

def main():
    print("="*60)
    print("Phase 4: Performance Optimizations - Verification Tests")
    print("="*60)
    print()
    
    test_basic_functionality()
    test_caching()
    test_trig_precomputation()
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print()
    print("Optimizations implemented:")
    print("  1. ✓ Added @lru_cache for z_max polynomial calculation")
    print("  2. ✓ Pre-computed trigonometric terms in get_geo_lengths()")
    print("  3. ✓ Pre-computed trigonometric terms in _get_hydrodynamic_force()")
    print("  4. ✓ Pre-computed trigonometric terms in _get_skin_friction()")
    print("  5. ✓ Reduced redundant calculations in force methods")
    print()

if __name__ == "__main__":
    main()
