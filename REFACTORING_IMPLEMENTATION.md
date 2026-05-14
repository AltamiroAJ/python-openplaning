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

## Phase 2: Remove pkg_resources & Pre-load Tables (NEXT)

### 2.1 Replace pkg_resources with importlib.resources
**Action Required:**
- Replace lines 6 and 1053-1062 in openplaning.py
- Use `importlib.resources` for Python 3.9+ compatibility
- Fallback for older Python versions

### 2.2 Embed CSV Tables as Constants
**Files to Convert:**
- Raw_0.2.csv (25 data rows)
- Raw_0.4.csv (25 data rows)
- Raw_0.6.csv (25 data rows)
- V_0.2.csv (25 data rows)
- V_0.4.csv (25 data rows)
- Raw_V_0.2.csv (75 data rows)
- Raw_V_0.4.csv (75 data rows)
- Raw_V_0.6.csv (75 data rows)

**Approach:**
- Convert each CSV to numpy array constants
- Store in separate `tables_data.py` module or at module top
- Load once during module initialization

## Phase 3: Extract Nested Functions

### 3.1 Functions to Extract from get_forces()
Lines 489-722 contain these nested functions:
1. `get_hydrodynamic_force()` (~40 lines)
2. `get_skin_friction()` (~70 lines)
3. `get_lift_change()` (~35 lines)
4. `get_air_resistance()` (~30 lines)
5. `get_flap_force()` (~25 lines)
6. `sum_forces()` (~30 lines)

**Action:** Convert to private methods:
- `_calculate_hydrodynamic_force()`
- `_calculate_skin_friction()`
- `_calculate_lift_change()`
- `_calculate_air_resistance()`
- `_calculate_flap_force()`
- `_sum_forces()`

### 3.2 Functions to Extract from get_eom_matrices()
Lines 798-867 contain nested functions:
1. `get_mass_matrix()` (~40 lines)
2. `get_damping_matrix()` (~30 lines)
3. `get_restoring_matrix()` (~30 lines)

**Action:** Convert to private methods:
- `_calculate_mass_matrix()`
- `_calculate_damping_matrix()`
- `_calculate_restoring_matrix()`

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
2. **NEXT:** Replace pkg_resources usage in openplaning.py
3. **NEXT:** Embed CSV tables as constants
4. Extract nested functions to private methods
5. Add pre-allocated arrays to __init__
6. Implement interpolation caching
7. Add trigonometric caching
8. Create dataclass configuration
9. Define named constants
10. Implement result objects

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
- ⏳ `/workspace/openplaning/openplaning.py` - Pending updates
- ⏳ `/workspace/openplaning/tables_data.py` - To be created
- ⏳ `/workspace/openplaning/config.py` - To be created (dataclasses)
- ⏳ `/workspace/openplaning/results.py` - To be created (result objects)
