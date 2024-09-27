"""
Adaption of code from https://github.com/dottormale/Qtransform
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class SplineInterpolateBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_knots_and_basis_matrices(self, x, k, s):
        knots = self.generate_natural_knots(x, k)
        basis_matrix = self.bspline_basis_natural(x, k, knots)
        identity = torch.eye(basis_matrix.shape[-1])
        B_T_B = basis_matrix.T @ basis_matrix + s * identity
        return knots, basis_matrix, B_T_B

    def generate_natural_knots(self, x: Tensor, k: int) -> Tensor:
        """
        Generates a natural knot sequence for B-spline interpolation.
        Natural knot sequence means that 2*k knots are added to the beginning
        and end of datapoints as replicas of first and last datapoint
        respectively in order to enforce natural boundary conditions,
        i.e. second derivative = 0.
        The other n nodes are placed in correspondece of the data points.

        Args:
            x: Tensor of data point positions.
            k: Degree of the spline.

        Returns:
            Tensor of knot positions.
        """
        return F.pad(x[None], (k, k), mode="replicate").squeeze(0)

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
    def __init__(
        self,
        k=3,
        s=0.001,
        x_in: Optional[Tensor] = None,
        x_out: Optional[Tensor] = None,
    ):
        super().__init__()
        self.k = k
        self.s = s
        self.register_buffer("x_in", x_in)
        self.register_buffer("x_out", x_out)

        if self.x_in is not None:
            tx, Bx, BxT_Bx = self._compute_knots_and_basis_matrices(x_in, k, s)
            self.register_buffer("t", tx)
            self.register_buffer("B", Bx)
            self.register_buffer("B_T_B", BxT_Bx)

        if self.x_out is not None:
            if self.x_in is None:
                raise ValueError(
                    "If x_out is specified, x_in must also be given"
                )
            Bx_out = self.bspline_basis_natural(x_out, k, self.tx)
            self.register_buffer("Bx_out", Bx_out)

    def univariate_spline_fit_natural(self, Z):
        B_T_z = self.Bx.transpose(-2, -1) @ Z.unsqueeze(
            -1
        )  # (batch_size, m, 1)
        # Solve the linear system for each batch
        coef = torch.linalg.solve(
            self.BxT_Bx.expand(Z.size(0), -1, -1), B_T_z
        ).squeeze(-1)
        return coef

    def evaluate_univariate_spline(self, C: Tensor):
        """
        Evaluate a bivariate spline on a grid of x and y points.

        Args:
            C: Coefficient tensor of shape (batch_size, mx, my).

        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        # Perform batched matrix multiplication:
        # (batch_size, n, m) @ (batch_size, m, 1) -> (batch_size, n)
        return (self.Bx_out.unsqueeze(0) @ C.unsqueeze(-1)).squeeze(-1)

    def forward(
        self,
        Z: Tensor,
        x_in: Optional[Tensor] = None,
        x_out: Optional[Tensor] = None,
    ) -> Tensor:
        if x_out is None and self.x_out is None:
            raise ValueError(
                "Output x-coordinates were not specified in either object "
                "creation or in forward call"
            )

        if len(Z.shape) > 3:
            raise ValueError("Input data has more than 3 dimensions")

        while len(Z.shape) < 3:
            Z = Z.unsqueeze(0)

        nx_points = Z.shape[-1:]

        if self.x_in is None and x_in is None:
            x_in = torch.linspace(-1, 1, nx_points)

        if x_in is not None:
            self.x_in = x_in if x_in is not None else self.x_in
            (
                self.tx,
                self.Bx,
                self.BxT_Bx,
            ) = self._compute_knots_and_basis_matrices(
                self.x_in, self.kx, self.sx
            )
        if x_out is not None:
            self.x_out = x_out if x_out is not None else self.x_out
            self.Bx_out = self.bspline_basis_natural(
                self.x_out, self.kx, self.tx
            )

        coef = self.univariate_spline_fit_natural(Z)
        Z_interp = self.evaluateunivariate_spline(coef)
        return Z_interp


class SplineInterpolate2D(SplineInterpolateBase):
    def __init__(
        self,
        kx=3,
        ky=3,
        sx=0.001,
        sy=0.001,
        x_in: Optional[Tensor] = None,
        y_in: Optional[Tensor] = None,
        x_out: Optional[Tensor] = None,
        y_out: Optional[Tensor] = None,
        logf: Optional[bool] = False,
    ):
        super().__init__()
        self.kx = kx
        self.ky = ky
        self.sx = sx
        self.sy = sy
        self.logf = logf
        self.register_buffer("x_in", x_in)
        self.register_buffer("y_in", y_in)
        self.register_buffer("x_out", x_out)
        self.register_buffer("y_out", y_out)

        if self.x_in is not None:
            tx, Bx, BxT_Bx = self._compute_knots_and_basis_matrices(
                x_in, kx, sx
            )
            self.register_buffer("tx", tx)
            self.register_buffer("Bx", Bx)
            self.register_buffer("BxT_Bx", BxT_Bx)
        if self.y_in is not None:
            ty, By, ByT_By = self._compute_knots_and_basis_matrices(
                y_in, ky, sy
            )
            self.register_buffer("ty", ty)
            self.register_buffer("By", By)
            self.register_buffer("ByT_By", ByT_By)

        if self.x_out is not None:
            if self.x_in is None:
                raise ValueError(
                    "If x_out is specified, x_in must also be given"
                )
            Bx_out = self.bspline_basis_natural(x_out, kx, self.tx)
            self.register_buffer("Bx_out", Bx_out)
        if self.y_out is not None:
            if self.y_in is None:
                raise ValueError(
                    "If y_out is specified, y_in must also be given"
                )
            By_out = self.bspline_basis_natural(y_out, ky, self.ty)
            self.register_buffer("By_out", By_out)

    def bivariate_spline_fit_natural(self, Z):

        # Adding batch dimension handling
        ByT_Z_Bx = (
            torch.einsum("ij,bcjk->bcik", self.By.T, Z.transpose(-2, -1))
            @ self.Bx
        )  # (batch, channel, my, mx)
        E = torch.linalg.solve(self.ByT_By, ByT_Z_Bx)  # (batch_size, my, mx)
        C = torch.linalg.solve(self.BxT_Bx, E.transpose(-2, -1)).transpose(
            -2, -1
        )  # (batch_size, channel, mx, my)

        return C

    def evaluate_bivariate_spline(self, C: Tensor):
        """
        Evaluate a bivariate spline on a grid of x and y points.

        Args:
            C: Coefficient tensor of shape (batch_size, mx, my).

        Returns:
            Z_interp: Interpolated values at the grid points.
        """
        # Perform matrix multiplication using einsum to get Z_interp
        return torch.einsum(
            "ik,bckm,mj->bcij", self.By_out, C, self.Bx_out.transpose(-2, -1)
        )

    def forward(
        self,
        Z: Tensor,
        x_in: Optional[Tensor] = None,
        y_in: Optional[Tensor] = None,
        x_out: Optional[Tensor] = None,
        y_out: Optional[Tensor] = None,
    ) -> Tensor:
        if x_out is None and self.x_out is None:
            raise ValueError(
                "Output x-coordinates were not specified in either object "
                "creation or in forward call"
            )

        if y_out is None and self.y_out is None:
            raise ValueError(
                "Output x-coordinates were not specified in either object "
                "creation or in forward call"
            )

        if len(Z.shape) > 4:
            raise ValueError("Input data has more than 4 dimensions")

        while len(Z.shape) < 4:
            Z = Z.unsqueeze(0)

        ny_points, nx_points = Z.shape[-2:]

        if self.x_in is None and x_in is None:
            x_in = torch.linspace(-1, 1, nx_points)
        if self.y_in is None and y_in is None:
            if self.logf:
                y_in = torch.logspace(-1, 1, ny_points)
            else:
                y_in = torch.linspace(-1, 1, ny_points)

        if x_in is not None:
            self.x_in = x_in if x_in is not None else self.x_in
            (
                self.tx,
                self.Bx,
                self.BxT_Bx,
            ) = self._compute_knots_and_basis_matrices(
                self.x_in, self.kx, self.sx
            )
        if y_in is not None:
            self.y_in = y_in if y_in is not None else self.y_in
            (
                self.ty,
                self.By,
                self.ByT_By,
            ) = self._compute_knots_and_basis_matrices(
                self.y_in, self.ky, self.sy
            )
        if x_out is not None:
            self.x_out = x_out if x_out is not None else self.x_out
            self.Bx_out = self.bspline_basis_natural(
                self.x_out, self.kx, self.tx
            )
        if y_out is not None:
            self.y_out = y_out if y_out is not None else self.y_out
            self.By_out = self.bspline_basis_natural(
                self.y_out, self.ky, self.ty
            )

        coef = self.bivariate_spline_fit_natural(Z)
        Z_interp = self.evaluate_bivariate_spline(coef)
        return Z_interp
