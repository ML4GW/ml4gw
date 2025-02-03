import numpy as np
from scipy.optimize import root_scalar as scipy_root_scalar

from ml4gw.optimize import root_scalar


def poly(x):
    return x**5 - 4 * x**4 + 3 * x**3 - 2 * x**2 + x - 1


def poly_deriv(x):
    return 5 * x**4 - 16 * x**3 + 9 * x**2 - 4 * x + 1


def test_root_scalar():
    tol, x0 = 1e-6, 5.0

    result = root_scalar(f=poly, fprime=poly_deriv, xtol=tol, x0=x0)
    assert result["converged"]
    assert np.isclose(poly(result["root"]), 0, atol=tol)

    scipy_result = scipy_root_scalar(
        f=poly, fprime=poly_deriv, xtol=tol, x0=x0
    )
    assert scipy_result.converged
    assert np.isclose(poly(scipy_result.root), 0, atol=tol)

    assert np.isclose(result["root"], scipy_result.root, atol=tol)
