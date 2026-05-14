"""
Numerical differentiation and root-finding methods.

This module provides implementations of:
- Finite difference gradient calculation
- Complex step gradient calculation  
- N-dimensional Newton-Raphson root finding

These are embedded here to remove external dependencies for Grasshopper compatibility.
"""

import numpy as np


def finiteGrad(func, x0, step):
    """
    Calculate the Jacobian of a function using forward finite differences.
    
    Parameters
    ----------
    func : callable
        Function to perform numerical differentiation on.
        Should accept array-like input and return array-like output.
    x0 : array_like
        Location to calculate gradient of function.
    step : float
        Step size for finite difference approximation.
    
    Returns
    -------
    ndarray
        Jacobian matrix of shape (n, m) where n is output dimension 
        and m is input dimension.
    
    Examples
    --------
    >>> finiteGrad(lambda x: x**2, [1], 1e-7)
    array([[2.]])
    """
    x0 = np.array(x0, dtype=float)
    xLength = len(x0)
    fVal = np.array(func(x0), dtype=float)
    h = step
    hMatrix = np.eye(xLength) * h
    
    jacobian = np.zeros((len(fVal), xLength))
    for n in range(xLength):
        x_perturbed = x0 + hMatrix[n, :]
        f_perturbed = np.array(func(x_perturbed), dtype=float)
        jacobian[:, n] = (f_perturbed - fVal) / h
    
    return jacobian


def complexGrad(func, x0):
    """
    Calculate the Jacobian using complex step differentiation.
    
    This method provides higher accuracy than finite differences and
    avoids subtractive cancellation errors.
    
    Parameters
    ----------
    func : callable
        Function to perform complex differentiation on.
        MUST be compatible with complex arithmetic (no operations that
        discard imaginary parts).
    x0 : array_like
        Location to calculate gradient of function.
    
    Returns
    -------
    ndarray
        Jacobian matrix of shape (n, m) where n is output dimension
        and m is input dimension.
    
    Examples
    --------
    >>> complexGrad(lambda x: x**2, [1])
    array([[2.]])
    
    Notes
    -----
    The function MUST properly handle complex numbers. Common issues:
    - Using np.real() or .real which discards imaginary parts
    - Comparisons with complex numbers
    - Some numpy functions that don't support complex dtype
    """
    x0 = np.array(x0, dtype=float)
    xLength = len(x0)
    h = 1e-30  # Arbitrarily small step, no need to modify
    
    # Determine output dimension by evaluating function once
    f_val = func(x0)
    f_dim = len(np.asarray(f_val))
    
    jacobian = np.zeros((f_dim, xLength))
    hMatrix = np.eye(xLength) * h
    
    for n in range(xLength):
        x_perturbed = x0 + 1j * hMatrix[n, :]
        f_perturbed = np.array(func(x_perturbed), dtype=complex)
        jacobian[:, n] = np.imag(f_perturbed) / h
    
    return jacobian


def nDimNewton(func, x0, fprime, tol=1e-6, maxiter=50, xlim=None, 
               heh=True, hehcon=None):
    """
    N-dimensional Newton-Raphson root finding method.
    
    Finds x such that func(x) = 0 using Newton's method with optional
    heuristic error handling and bounds checking.
    
    Parameters
    ----------
    func : callable
        N-dimensional function to find root for, where n > 1.
        Should return array-like output.
    x0 : array_like
        Initial estimate for the root.
    fprime : callable
        Function returning the Jacobian matrix of func.
        Signature: fprime(x) -> ndarray of shape (n, m).
    tol : float, optional
        Tolerance for convergence (default: 1e-6).
        Convergence achieved when ||func(x)|| < tol.
    maxiter : int, optional
        Maximum number of iterations (default: 50).
    xlim : array_like, optional
        Inclusive bounds for variables during iteration.
        Shape: ((x0_lower, x0_upper), (x1_lower, x1_upper), ...)
        Used to keep iterations in valid domain, not to constrain solution.
    heh : bool, optional
        Enable heuristic error handling (default: True).
        Includes rank checking and bisection backtracking.
    hehcon : callable, optional
        Constraint function for heuristic error handling.
        Should return array-like values where all must be <= 0.
        Used to maintain valid domain during iterations.
    
    Returns
    -------
    ndarray
        Approximate root of func(x) = 0.
    
    Raises
    ------
    RuntimeError
        If initial estimate doesn't satisfy constraints,
        if Jacobian is rank-deficient,
        if solution diverges from bounds,
        or if maximum iterations exceeded.
    
    Examples
    --------
    >>> def func(x):
    ...     return [(x[1]-3)**2 + x[0] - 5, 2*x[1] + x[0]**3]
    >>> def fprime(x):
    ...     return complexGrad(func, x)
    >>> x0 = [0, 0]
    >>> root = nDimNewton(func, x0, fprime)
    """
    # Initialize values
    k = 1  # Iteration count
    x = np.array(x0, dtype=float)
    xOld = None
    f = np.array(func(x), dtype=float)
    
    # Check initial constraint satisfaction
    if heh and hehcon is not None:
        con_val = np.asarray(hehcon(x))
        if any(con_val > 0):
            raise RuntimeError(
                "Initial estimate x0 does not satisfy hehcon(x0)[:]<=0. "
                "Try a different x0."
            )
    
    while np.linalg.norm(f) > tol:
        Df = fprime(x)
        
        # Check and correct rank deficiency with heuristic handling
        if heh:
            while np.linalg.matrix_rank(Df) < len(Df):
                if xOld is not None:
                    # Bisection backtracking
                    xNew = (x + xOld) / 2
                    x = xNew
                    Df = fprime(x)
                else:
                    raise RuntimeError(
                        "Initial estimate x0 does not produce a full rank Jacobian. "
                        "Try a different x0."
                    )
        
        xOld = x.copy()
        g = np.array(func(x), dtype=float)
        
        # Solve linear system Df @ v = -g
        try:
            v = np.linalg.solve(Df, -g)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Linear solve failed: {str(e)}")
        
        # Check bounds violations before updating
        if xlim is not None:
            xlim = np.asarray(xlim)
            runaways_lower = [
                f'x[{i}]' for i in range(len(x)) 
                if x[i] <= xlim[i, 0] and v[i] < 0
            ]
            runaways_upper = [
                f'x[{i}]' for i in range(len(x)) 
                if x[i] >= xlim[i, 1] and v[i] > 0
            ]
            
            if runaways_lower or runaways_upper:
                message = (
                    'No solution found inside constraints. Iteration stopped because '
                    'the following variables were found outside user-defined limits:\n'
                )
                if runaways_lower:
                    message += ', '.join(runaways_lower) + ' <= lower bound\n'
                if runaways_upper:
                    message += ', '.join(runaways_upper) + ' >= upper bound\n'
                raise RuntimeError(message)
        
        # Update x
        x = x + v
        
        # Force constraints if enabled
        if heh and xlim is not None:
            for i in range(len(x)):
                if x[i] < xlim[i, 0]:
                    x[i] = xlim[i, 0]
                elif x[i] > xlim[i, 1]:
                    x[i] = xlim[i, 1]
        
        # Check iteration limit
        if k == maxiter:
            raise RuntimeError(
                'Solution did not converge after maximum number of iterations.'
            )
        
        k += 1
        f = np.array(func(x), dtype=float)
        
        # Enforce heuristic constraints with backtracking
        if heh and hehcon is not None:
            while any(np.asarray(hehcon(x)) > 0):
                # Bisection backtracking
                x = (x + xOld) / 2
                f = np.array(func(x), dtype=float)
    
    return x
