"""
Adaptation of code from https://github.com/dottormale/Qtransform_torch/
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


class SplineInterpolateBase(torch.nn.Module):
    """
    Base class for spline interpolation.
    """

    def _compute_knots_and_basis_matrices(self, x, k, s):
        knots = self.generate_fitpack_knots(x, k)
        basis_matrix = self.bspline_basis_natural(x, k, knots)
        identity = torch.eye(basis_matrix.shape[-1])
        B_T_B = basis_matrix.T @ basis_matrix + s * identity
        return knots, basis_matrix, B_T_B

    def generate_fitpack_knots(self, x: Tensor, k: int) -> Tensor:
        """
        Generates a knot sequence for B-spline interpolation
        in the same way as the FITPACK algorithm used by SciPy.

        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.

        Returns:
            Tensor of knot positions.
        """
        num_knots = x.shape[-1] + k + 1
        knots = torch.zeros(num_knots, dtype=x.dtype)
        knots[: k + 1] = x[0]
        knots[-(k + 1) :] = x[-1]

        # Interior knots are the rolling average of the data points
        # excluding the first and last points
        windows = x[1:-1].unfold(dimension=-1, size=k, step=1)
        knots[k + 1 : -k - 1] = windows.mean(dim=-1)

        return knots

    def compute_L_R(
        self,
        x: Tensor,
        t: Tensor,
        d: int,
        m: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the L and R values for B-spline basis functions.
        L and R are respectively the first and second coefficient multiplying
        B_{i,p-1}(x) and B_{i+1,p-1}(x) in De Boor's recursive formula for
        Bspline basis funciton computation
        See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for details

        Args:
            x:
                Tensor of data point positions.
            t:
                Tensor of knot positions.
            d:
                Current degree of the basis function.
            m:
                Number of intervals (n - k - 1, where n is the number of knots
                and k is the degree).

        Returns:
            L: Tensor containing left values for the B-spline basis functions.
            R: Tensor containing right values for the B-spline basis functions.
        """
        left_num = x.unsqueeze(1) - t[:m].unsqueeze(0)
        left_den = t[d : m + d] - t[:m]
        L = left_num / left_den.unsqueeze(0)
        L = torch.nan_to_num_(L, nan=0.0, posinf=0.0, neginf=0.0)

        right_num = t[d + 1 : m + d + 1] - x.unsqueeze(1)
        right_den = t[d + 1 : m + d + 1] - t[1 : m + 1]
        R = right_num / right_den.unsqueeze(0)
        R = torch.nan_to_num_(R, nan=0.0, posinf=0.0, neginf=0.0)

        return L, R

    def zeroth_order(
        self,
        x: Tensor,
        k: int,
        t: Tensor,
        n: int,
        m: int,
    ) -> Tensor:
        """
        Compute the zeroth-order B-spline basis functions
        according to de Boors recursive formula.
        See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference

        Args:
            x:
                Tensor of data point positions.
            k:
                Degree of the spline.
            t:
                Tensor of knot positions.
            n:
                Number of data points.
            m:
                Number of intervals (n - k - 1, where n is the number of knots
                and k is the degree).

        Returns:
            b: Tensor containing the zeroth-order B-spline basis functions.
        """
        b = torch.zeros((n, m, k + 1))

        mask_lower = t[: m + 1].unsqueeze(0)[:, :-1] <= x.unsqueeze(1)
        mask_upper = x.unsqueeze(1) < t[: m + 1].unsqueeze(0)[:, 1:]

        b[:, :, 0] = mask_lower & mask_upper
        b[:, 0, 0] = torch.where(x < t[1], torch.ones_like(x), b[:, 0, 0])
        b[:, -1, 0] = torch.where(x >= t[-2], torch.ones_like(x), b[:, -1, 0])
        return b

    def bspline_basis_natural(
        self,
        x: Tensor,
        k: int,
        t: Tensor,
    ) -> Tensor:
        """
        Compute bspline basis function using de Boor's recursive formula
        (See https://en.wikipedia.org/wiki/De_Boor%27s_algorithm for reference)
        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.
            t: Tensor of knot positions.

        Returns:
            Tensor containing the kth-order B-spline basis functions
        """
        n = x.shape[0]
        m = t.shape[0] - k - 1

        # calculate zeroth order basis funciton
        b = self.zeroth_order(x, k, t, n, m)

        zeros_tensor = torch.zeros(b.shape[0], 1)
        # recursive de Boors formula for bspline basis functions
        for d in range(1, k + 1):
            L, R = self.compute_L_R(x, t, d, m)
            left = L * b[:, :, d - 1]

            temp_b = torch.cat([b[:, 1:, d - 1], zeros_tensor], dim=1)

            right = R * temp_b
            b[:, :, d] = left + right

        return b[:, :, -1]


class SplineInterpolate1D(SplineInterpolateBase):
    """
    Perform 1D spline interpolation based on De Boor's method.
    It is allowed to have two spatial dimensions, but the second
    dimension cannot be interpolated along. To interpolate along both
    dimensions, use :class:`SplineInterpolate2D`.

    Supports batched, multi-channel inputs, so acceptable data
    shapes are ``(width)``, ``(height, width)``, ``(batch, width)``,
    ``(batch, height, width)``, ``(batch, channel, width)``, and
    ``(batch, channel, height, width)``.

    During initialization of this Module, both the desired input
    and output coordinate Tensors can be specified to allow
    pre-computation of the B-spline basis matrices, though the only
    mandatory argument is the coordinates of the data along the
    ``width`` dimension.

    Unlike scipy's implementation of spline interpolation, the data
    to be interpolated is not passed until actually calling the
    object. This is useful for cases where the input and output
    coordinates are known in advance, but the data is not, so that
    the interpolator can be set up ahead of time.

    Args:
        x_in:
            Coordinates of the width dimension of the data
        kx:
            Degree of spline interpolation along the width dimension.
            Default is cubic.
        sx:
            Regularization factor to avoid singularities during matrix
            inversion for interpolation along the width dimension. Not
            to be confused with the ``s`` parameter in scipy's spline
            methods, which controls the number of knots.
        x_out:
            Coordinates for the data to be interpolated to along the
            width dimension. If not specified during initialization,
            this must be specified during the object call.

    """

    def __init__(
        self,
        x_in: Tensor,
        kx: int = 3,
        sx: float = 0.0,
        x_out: Optional[Tensor] = None,
    ):
        super().__init__()

        if len(x_in) < kx + 2:
            raise ValueError(
                "Input x-coordinates must have at least kx + 2 points."
            )

        # Ensure that coordinates are floats
        x_in = x_in.float()
        x_out = x_out.float() if x_out is not None else None

        self.kx = kx
        self.sx = sx
        self.register_buffer("x_in", x_in)
        self.register_buffer("x_out", x_out)

        tx, Bx, BxT_Bx = self._compute_knots_and_basis_matrices(x_in, kx, sx)
        self.register_buffer("tx", tx)
        self.register_buffer("Bx", Bx)
        self.register_buffer("BxT_Bx", BxT_Bx)

        if self.x_out is not None:
            x_clamped = torch.clamp(x_out, tx[kx], tx[-kx - 1])
            Bx_out = self.bspline_basis_natural(x_clamped, kx, self.tx)
            self.register_buffer("Bx_out", Bx_out)

    def spline_fit_natural(self, Z):
        # Adding batch/channel dimension handling
        # Bx @ Z
        BxT_Z = torch.einsum("ij,bchj->bchi", self.Bx.T, Z)
        # (BxT @ Bx)^-1 @ (BxT @ Z) = Bx^-1 @ Z
        C = torch.linalg.solve(self.BxT_Bx, BxT_Z.unsqueeze(-1))
        return C.squeeze(-1)

    def evaluate_spline(self, C: Tensor):
        """
        Evaluate a bivariate spline on a grid of x and y points.

        Args:
            C: Coefficient tensor of shape (batch_size, mx, my).

        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        # Perform matrix multiplication using einsum to get Z_interp
        return torch.einsum("ij,bchj->bchi", self.Bx_out, C)

    def _validate_inputs(self, Z, x_out):
        if x_out is None and self.x_out is None:
            raise ValueError(
                "Output x-coordinates were not specified in either object "
                "creation or in forward call"
            )

        dims = len(Z.shape)
        if dims > 4:
            raise ValueError("Input data has more than 4 dimensions")

        if Z.shape[-1] != len(self.x_in):
            raise ValueError(
                "The spatial dimensions of the data tensor do not match "
                "the given input dimensions. "
                f"Expected {len(self.x_in)}, but got {Z.shape[-1]}"
            )

        # Expand Z to have a batch, channel, and height dimension if needed
        while len(Z.shape) < 4:
            Z = Z.unsqueeze(0)

        return Z

    def forward(
        self,
        Z: Tensor,
        x_out: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the interpolated data

        Args:
            Z:
                Tensor of data to be interpolated. Must be between 2 and 4
                dimensions. The shape of the tensor must agree with the
                input coordinates given on initialization.
            x_out:
                Coordinates to interpolate the data to along the width
                dimension. Overrides any value that was set during
                initialization.

        Returns:
            A 4D tensor with shape ``(batch, channel, height, width)``.
            Depending on the input data shape, many of these dimensions
            may have length 1.
        """

        Z = self._validate_inputs(Z, x_out)

        if x_out is not None:
            x_out = x_out.float()
            x_clamped = torch.clamp(
                x_out, self.tx[self.kx], self.tx[-self.kx - 1]
            )
            self.Bx_out = self.bspline_basis_natural(
                x_clamped, self.kx, self.tx
            )

        coef = self.spline_fit_natural(Z)
        Z_interp = self.evaluate_spline(coef)
        return Z_interp


class SplineInterpolate2D(SplineInterpolateBase):
    """
    Perform 2D spline interpolation based on De Boor's method.
    Supports batched, multi-channel inputs, so acceptable data
    shapes are ``(height, width)``, ``(batch, height, width)``,
    and ``(batch, channel, height, width)``.

    During initialization of this Module, both the desired input
    and output coordinate Tensors can be specified to allow
    pre-computation of the B-spline basis matrices, though the only
    mandatory arguments are the input coordinates.

    Unlike scipy's implementation of spline interpolation, the data
    to be interpolated is not passed until actually calling the
    object. This is useful for cases where the input and output
    coordinates are known in advance, but the data is not, so that
    the interpolator can be set up ahead of time.

    Args:
        x_in:
            Coordinates of the width dimension of the data
        y_in:
            Coordinates of the height dimension of the data.
        kx:
            Degree of spline interpolation along the width dimension.
            Default is cubic.
        ky:
            Degree of spline interpolation along the height dimension.
            Default is cubic.
        sx:
            Regularization factor to avoid singularities during matrix
            inversion for interpolation along the width dimension. Not
            to be confused with the ``s`` parameter in scipy's spline
            methods, which controls the number of knots.
        sy:
            Regularization factor to avoid singularities during matrix
            inversion for interpolation along the height dimension.
        x_out:
            Coordinates for the data to be interpolated to along the
            width dimension. If not specified during initialization,
            this must be specified during the object call.
        y_out:
            Coordinates for the data to be interpolated to along the
            height dimension. If not specified during initialization,
            this must be specified during the object call.

    """

    def __init__(
        self,
        x_in: Tensor,
        y_in: Tensor,
        kx: int = 3,
        ky: int = 3,
        sx: float = 0.0,
        sy: float = 0.0,
        x_out: Optional[Tensor] = None,
        y_out: Optional[Tensor] = None,
    ):
        super().__init__()

        if len(x_in) < kx + 2:
            raise ValueError(
                "Input x-coordinates must have at least kx + 2 points."
            )
        if len(y_in) < ky + 2:
            raise ValueError(
                "Input y-coordinates must have at least ky + 2 points."
            )

        # Ensure that coordinates are floats
        x_in = x_in.float()
        y_in = y_in.float()
        x_out = x_out.float() if x_out is not None else None
        y_out = y_out.float() if y_out is not None else None

        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.register_buffer("x_in", x_in)
        self.register_buffer("y_in", y_in)
        self.register_buffer("x_out", x_out)
        self.register_buffer("y_out", y_out)

        tx, Bx, BxT_Bx = self._compute_knots_and_basis_matrices(x_in, kx, sx)
        self.register_buffer("tx", tx)
        self.register_buffer("Bx", Bx)
        self.register_buffer("BxT_Bx", BxT_Bx)

        ty, By, ByT_By = self._compute_knots_and_basis_matrices(y_in, ky, sy)
        self.register_buffer("ty", ty)
        self.register_buffer("By", By)
        self.register_buffer("ByT_By", ByT_By)

        if self.x_out is not None:
            x_clamped = torch.clamp(x_out, tx[kx], tx[-kx - 1])
            Bx_out = self.bspline_basis_natural(x_clamped, kx, self.tx)
            self.register_buffer("Bx_out", Bx_out)
        if self.y_out is not None:
            y_clamped = torch.clamp(y_out, ty[ky], ty[-ky - 1])
            By_out = self.bspline_basis_natural(y_clamped, ky, self.ty)
            self.register_buffer("By_out", By_out)

    def bivariate_spline_fit_natural(self, Z):
        # Adding batch/channel dimension handling
        # ByT @ Z @ BxW
        ByT_Z_Bx = torch.einsum("ij,bcik,kl->bcjl", self.By, Z, self.Bx)
        # (ByT @ By)^-1 @ (ByT @ Z @ Bx) = By^-1 @ Z @ Bx
        E = torch.linalg.solve(self.ByT_By, ByT_Z_Bx)
        # ((BxT @ Bx)^-1 @ (By^-1 @ Z @ Bx)T)T = By^-1 @ Z @ BxT^-1
        return torch.linalg.solve(self.BxT_Bx, E.mT).mT

    def evaluate_bivariate_spline(self, C: Tensor):
        """
        Evaluate a bivariate spline on a grid of x and y points.

        Args:
            C: Coefficient tensor of shape (batch_size, mx, my).

        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        # Perform matrix multiplication using einsum to get Z_interp
        return torch.einsum("ik,bckm,mj->bcij", self.By_out, C, self.Bx_out.mT)

    def _validate_inputs(self, Z, x_out, y_out):
        if x_out is None and self.x_out is None:
            raise ValueError(
                "Output x-coordinates were not specified in either object "
                "creation or in forward call"
            )

        if y_out is None and self.y_out is None:
            raise ValueError(
                "Output y-coordinates were not specified in either object "
                "creation or in forward call"
            )

        dims = len(Z.shape)
        if dims > 4:
            raise ValueError("Input data has more than 4 dimensions")
        if dims < 2:
            raise ValueError("Input data has fewer than 2 dimensions")

        if Z.shape[-2:] != torch.Size([len(self.y_in), len(self.x_in)]):
            raise ValueError(
                "The spatial dimensions of the data tensor do not match "
                "the given input dimensions. "
                f"Expected [{len(self.y_in)}, {len(self.x_in)}], but got "
                f"[{Z.shape[-2]}, {Z.shape[-1]}]"
            )

        # Expand Z to have a batch and channel dimension if needed
        while len(Z.shape) < 4:
            Z = Z.unsqueeze(0)

        return Z, y_out

    def forward(
        self,
        Z: Tensor,
        x_out: Optional[Tensor] = None,
        y_out: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the interpolated data

        Args:
            Z:
                Tensor of data to be interpolated. Must be between 2 and 4
                dimensions. The shape of the tensor must agree with the
                input coordinates given on initialization.
            x_out:
                Coordinates to interpolate the data to along the width
                dimension. Overrides any value that was set during
                initialization.
            y_out:
                Coordinates to interpolate the data to along the height
                dimension. Overrides any value that was set during
                initialization.

        Returns:
            A 4D tensor with shape ``(batch, channel, height, width)``.
            Depending on the input data shape, many of these dimensions
            may have length 1.
        """

        Z, y_out = self._validate_inputs(Z, x_out, y_out)

        if x_out is not None:
            x_out = x_out.float()
            x_clamped = torch.clamp(
                x_out, self.tx[self.kx], self.tx[-self.kx - 1]
            )
            self.Bx_out = self.bspline_basis_natural(
                x_clamped, self.kx, self.tx
            )
        if y_out is not None:
            y_out = y_out.float()
            y_clamped = torch.clamp(
                y_out, self.ty[self.ky], self.ty[-self.ky - 1]
            )
            self.By_out = self.bspline_basis_natural(
                y_clamped, self.ky, self.ty
            )

        coef = self.bivariate_spline_fit_natural(Z)
        Z_interp = self.evaluate_bivariate_spline(coef)
        return Z_interp
