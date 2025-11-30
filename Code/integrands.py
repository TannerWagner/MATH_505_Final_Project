import numpy as np
from math import exp, sqrt
from math import erf

def f_A(x):
    """
    Symmetric exponential integrand.
    x: array of shape (..., d)
    returns: array of shape (...)
    """
    return np.exp(-np.sum(x, axis=-1))

def I_A_exact(d):
    """
    Exact integral of f_A over [0,1]^d: (1 - e^{-1})^d.
    """
    return (1.0 - exp(-1.0))**d

def f_B(x):
    """
    Rational integrand: prod_j 1 / (1 + j x_j^2).
    x: array of shape (..., d)
    """

    # Indices: j = 1,...,d
    d = x.shape[-1]
    j = np.arange(1, d+1)
    
    return np.prod(1.0 / (1.0 + j * x**2), axis=-1)

def I_B_exact(d):
    """
    Exact integral of f_B over [0,1]^d:
    prod_{j=1}^d (1/sqrt(j)) * arctan(sqrt(j)).
    """
    j = np.arange(1, d+1, dtype=float)
    
    return np.prod((1.0 / np.sqrt(j)) * np.arctan(np.sqrt(j)))

def f_C(x):
    """
    Gaussian-type integrand: exp(-sum_j x_j^2).
    x: array of shape (..., d)
    """
    return np.exp(-np.sum(x**2, axis=-1))

def I_C_exact(d):
    """
    Exact integral of f_C over [0,1]^d:
    ( (sqrt(pi)/2) * erf(1) )^d.
    """
    one_d = 0.5 * np.sqrt(np.pi) * erf(1.0)
    
    return one_d**d

def f_D(x):
    """
    Low effective-dimension integrand:
    f_D(x) = ((x1 + ... + x5)/5)^2.

    Uses only the first 5 coordinates, even if d > 5.
    x: array of shape (..., d)
    returns: array of shape (...)
    """
    # Average over the first 5 coordinates
    s = np.mean(x[..., :5], axis=-1)
    
    return s**2

def I_D_exact(d):
    """
    Exact integral of f_D over [0,1]^d for d >= 5:
    I = 4/15, independent of d.
    """
    return 4.0 / 15.0
