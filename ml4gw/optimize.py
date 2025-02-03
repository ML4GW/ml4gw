from typing import Callable

import torch


def root_scalar(
    f: Callable,
    x0: float,
    args: tuple = (),
    fprime: Callable | None = None,
    maxiter: int = 100,
    xtol: float = 1e-6,
):
    """
    Find a root of a scalar function.

    Args:
        f (callable): The function whose root is to be found.
        x0 (float): Initial guess.
        args (tuple, optional): Extra arguments passed to the objective
        function `f` and its derivative(s).
        fprime (callable, optional): The derivative of the function.
        xtol (float, optional): The tolerance for the root.
        maxiter (int, optional): The maximum number of iterations.

    Returns:
        dict: A dictionary containing the root, and whether the optimization
        was successful.
    """
    if x0 is None:
        raise ValueError("x0 must be provided")
    res = {"converged": False, "root": None}
    for _ in range(maxiter):
        fx = f(x0, *args)
        if fprime is not None:
            fpx = fprime(x0, *args)
        else:
            fpx = (f(x0 + xtol, *args) - f(x0 - xtol, *args)) / (2 * xtol)
        if abs(fpx) < torch.finfo(torch.float).eps:
            res["root"] = x0
            res["converged"] = True
            return res
        x1 = x0 - fx / fpx
        if abs(x1 - x0) < xtol:
            res["root"] = x1
            res["converged"] = True
            return res
        x0 = x1
    return res
