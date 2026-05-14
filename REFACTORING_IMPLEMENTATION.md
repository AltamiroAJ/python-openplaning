# Refactoring Implementation Plan

## Overview
This document tracks the implementation of refactoring improvements to `openplaning.py` as outlined in `REFACTORING_ANALYSIS.md`.

## Phase 1: Critical Dependencies & Structure (COMPLETED)

### 1.1 Replace ndmath External Dependency
**Status:** ✅ COMPLETED
- Created `/workspace/ndmath.py` with inline implementations of:
  - `finiteGrad()` - Forward finite difference gradient
  - `complexGrad()` - Complex step gradient  
  - `nDimNewton()` - N-dimensional Newton-Raphson solver
- Improvements over original:
  - Better error handling for singular matrices
  - Explicit dtype specifications for numerical stability
  - Enhanced documentation with examples
  - Proper array copying to avoid mutation bugs

### 1.2 Evaluate ndmath.py for Further Refactoring
**Status:** ✅ ANALYZED

**Current State:** The ndmath.py file is well-structured with:
- Clear function separation
- Good documentation
- Proper error handling
- No external dependencies beyond numpy

**Refactoring Opportunities Identified:**
1. **Add Jacobian caching** - For repeated calls at same point (minor optimization)
2. **Add vectorized complexGrad** - Process multiple points simultaneously (future enhancement)
3. **Add convergence diagnostics** - Return iteration count, final residual (usability improvement)
4. **Add line search option** - Alternative to bisection backtracking (advanced feature)

**Decision:** Current implementation is sufficient for Phase 1. Enhancements can be added in Phase 4 if profiling shows bottlenecks.

### 1.3 Integration Strategy for openplaning.py
The ndmath.py module will be integrated into openplaning.py through:

```python
# In openplaning/openplaning.py
from ndmath import finiteGrad, complexGrad, nDimNewton

# Replace all ndmath.method() calls with direct function calls:
# ndmath.complexGrad(func, x) → complexGrad(func, x)
# ndmath.finiteGrad(func, x, h) → finiteGrad(func, x, h)
# ndmath.nDimNewton(...) → nDimNewton(...)
```

## Phase 2: Remove pkg_resources & Pre-load Tables (COMPLETED)

### 2.1 Replace pkg_resources with importlib.resources
**Status:** ✅ COMPLETED (Alternative approach: embedded tables directly)
- Removed `pkg_resources` import from line 6
- Replaced with import from `tables_data` module
- No file I/O required at runtime

### 2.2 Embed CSV Tables as Constants
**Status:** ✅ COMPLETED
- Created `/workspace/openplaning/tables_data.py` with all CSV data as numpy arrays
- Tables converted:
  - RAW_02_DATA, RAW_04_DATA, RAW_06_DATA (Raw resistance tables)
  - V_02_DATA, V_04_DATA (Velocity tables)
  - RAW_V_02_DATA, RAW_V_04_DATA, RAW_V_06_DATA (Combined tables)
  - Z_MAX_BETA_TABLE, Z_MAX_VALUES (z_max interpolation)
  - Z_MAX_POLY_COEFFS (polynomial fit)
- Updated `openplaning.py` to use embedded data (lines 1056-1066)
- Eliminated all `np.genfromtxt()` calls and file path resolution

**Benefits:**
- No external file dependencies
- Faster runtime (no file I/O)
- Works in isolated environments (Docker, Grasshopper, etc.)
- Simpler deployment (single package)

# Phase 3: Extract Nested Functions (COMPLETED)

### 3.1 Functions Extracted from get_forces()
**Status:** ✅ COMPLETED
- Converted nested functions to private methods:
  - `_get_hydrodynamic_force()` - Hydrodynamic force calculation
  - `_get_skin_friction()` - Skin friction calculation  
  - `_get_lift_change()` - Lift change due to roughness
  - `_get_air_resistance()` - Air drag estimation
  - `_get_flap_force()` - Flap force calculation
  - `_sum_forces()` - Orchestrates all force calculations

### 3.2 Functions Extracted from get_eom_matrices()
**Status:** ✅ COMPLETED
- Converted nested functions to private methods:
  - `_calculate_mass_matrix()` - Mass matrix coefficients
  - `_calculate_damping_matrix()` - Damping matrix coefficients
  - `_calculate_restoring_matrix()` - Restoring matrix coefficients

### 3.3 Additional Refinements in get_steady_trim()
**Status:** ✅ COMPLETED
- Renamed `_L_K` to `_L_K_constraint` for clarity

**Benefits Achieved:**
✓ Eliminated deeply nested function definitions
✓ Improved code testability and modularity
✓ Better separation of concerns
✓ Enhanced documentation with detailed docstrings
✓ Maintained full backward compatibility

**Verified Working:**
All methods tested successfully including:
- `get_forces()` with all extracted sub-methods
- `get_steady_trim()` with constraint function
- `get_eom_matrices()` with three new calculation methods
- `check_porpoising()` using EOM matrices
- `get_seaway_behavior()` with embedded tables

The refactored code produces identical results to the original implementation while being more maintainable and testable.

## Phase 4: Performance Optimizations

### 4.1 Cache Interpolation Functions
**Locations:**
- Line 324-325: z_max interpolation in get_geo_lengths()
- Lines 1080+: Multiple 2D interpolations in get_seaway_behavior()

**Solution:** Create InterpolationCache class with lazy-loaded interpolators

### 4.2 Pre-allocate Result Arrays
**Arrays to Pre-allocate:**
- hydrodynamic_force (3,)
- skin_friction (3,)
- lift_change (3,)
- air_resistance (3,)
- flap_force (3,)
- thrust_force (3,)
- net_force (3,)
- mass_matrix (2,2)
- damping_matrix (2,2)
- restoring_matrix (2,2)

### 4.3 Reduce Redundant Trig Calculations
**Strategy:**
- Add _deg_to_rad constant
- Cache sin/cos/tan values for beta, tau
- Update cache when angles change

## Phase 5: Usability Enhancements

### 5.1 Dataclass Configuration Pattern
Create `PlaningBoatConfig` dataclass with:
- All 28 constructor parameters
- Sensible defaults
- Input validation
- Builder pattern support

### 5.2 Named Constants for Magic Numbers
**Identified Magic Numbers:**
- Polynomial coefficients (line 322)
- Savitsky equation coefficients (lines 504-535)
- Z_max table values
- ITTC friction formula constants

### 5.3 Structured Result Objects
Create result classes:
- `ForceResult` - Contains forces + warnings
- `CalculationWarning` - Structured warning information
- `ValidityReport` - Parameter range checking

## Implementation Priority

1. ✅ **DONE:** Create ndmath.py with improved implementations
2. ✅ **DONE:** Replace pkg_resources usage in openplaning.py
3. ✅ **DONE:** Embed CSV tables as constants
4. ✅ **DONE:** Extract nested functions to private methods
5. ⏳ Add pre-allocated arrays to __init__
6. ⏳ Implement interpolation caching
7. ⏳ Add trigonometric caching
8. ⏳ Create dataclass configuration
9. ⏳ Define named constants
10. ⏳ Implement result objects

## Testing Strategy

Before each refactoring phase:
1. Run existing tests (if any)
2. Create baseline performance measurements
3. Verify output matches expected values

After each refactoring phase:
1. Run tests to ensure behavior unchanged
2. Measure performance improvements
3. Document any API changes

## Files Modified

- ✅ `/workspace/ndmath.py` - Created with numerical methods
- ✅ `/workspace/openplaning/openplaning.py` - Updated:
  - Removed `pkg_resources` import
  - Added import from `tables_data` module
  - Replaced file I/O with embedded data arrays (lines 1056-1066)
  - Extracted nested functions to private methods:
    - `_get_hydrodynamic_force()` (line 499)
    - `_get_skin_friction()` (line 539)
    - `_get_lift_change()` (line 605)
    - `_get_air_resistance()` (line 639)
    - `_get_flap_force()` (line 672)
    - `_sum_forces()` (line 699)
    - `_calculate_mass_matrix()` (line 764)
    - `_calculate_damping_matrix()` (line 819)
    - `_calculate_restoring_matrix()` (line 868)
  - Renamed constraint function in `get_steady_trim()` (line 753)
- ✅ `/workspace/openplaning/tables_data.py` - Created with all CSV data as numpy arrays
- ⏳ `/workspace/openplaning/config.py` - To be created (dataclasses)
- ⏳ `/workspace/openplaning/results.py` - To be created (result objects)
